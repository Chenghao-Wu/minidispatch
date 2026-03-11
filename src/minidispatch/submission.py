from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter
from hashlib import sha1

from tqdm import tqdm

from minidispatch._logging import log
from minidispatch.backend import Backend
from minidispatch.job import Job
from minidispatch.models import Resources, Task
from minidispatch.status import JobStatus


class Submission:
    """Orchestrates task execution: upload, submit, poll, recover, download."""

    def __init__(
        self,
        work_base: str,
        backend: Backend,
        resources: Resources,
        task_list: list[Task],
        forward_common_files: list[str] | None = None,
    ):
        self.work_base = work_base
        self.backend = backend
        self.resources = resources
        self.tasks = list(task_list)
        self.forward_common_files = sorted(forward_common_files or [])
        self.jobs: list[Job] = []
        self.backend.context.bind_submission(work_base, self.submission_hash)

    @property
    def submission_hash(self) -> str:
        return sha1(json.dumps(self.serialize(if_static=True)).encode()).hexdigest()

    def serialize(self, if_static: bool = False) -> dict:
        d: dict = {
            "work_base": self.work_base,
            "resources": self.resources.serialize(),
            "forward_common_files": self.forward_common_files,
            "belonging_jobs": [j.serialize(if_static=if_static) for j in self.jobs],
        }
        if not if_static:
            d["backend"] = self.backend.serialize()
        return d

    @classmethod
    def _from_dict(cls, d: dict) -> Submission:
        """Reconstruct from serialized dict (for recovery)."""
        backend = Backend.from_dict(d["backend"])
        resources = Resources.deserialize(d["resources"])
        # Reconstruct tasks from jobs
        jobs = [Job.deserialize(jd) for jd in d["belonging_jobs"]]
        tasks = [t for j in jobs for t in j.tasks]
        sub = cls.__new__(cls)
        sub.work_base = d["work_base"]
        sub.backend = backend
        sub.resources = resources
        sub.tasks = tasks
        sub.forward_common_files = d.get("forward_common_files", [])
        sub.jobs = jobs
        return sub

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Submission):
            return NotImplemented
        return json.dumps(self.serialize(if_static=True)) == json.dumps(
            other.serialize(if_static=True)
        )

    # --- Job generation ---

    def generate_jobs(self) -> None:
        if self.jobs:
            raise RuntimeError("Jobs already generated")
        task_num = len(self.tasks)
        if task_num == 0:
            raise RuntimeError("Must have at least 1 task")

        random.seed(42)
        indices = list(range(task_num))
        random.shuffle(indices)
        for idx in indices:
            self.jobs.append(
                Job(
                    tasks=[self.tasks[idx]],
                    resources=copy.deepcopy(self.resources),
                )
            )
        log.info(f"Generated {len(self.jobs)} jobs from {len(self.tasks)} tasks")

    # --- Recovery ---

    def try_recover(self) -> None:
        log.debug(f"Checking recovery file for {self.submission_hash}")
        fname = f"{self.submission_hash}.json"
        if not self.backend.context.check_file_exists(fname):
            return

        try:
            raw = self.backend.context.read_file(fname)
            data = json.loads(raw)
            recovered = Submission._from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning(f"Recovery JSON corrupted, starting fresh: {e}")
            return

        if self == recovered:
            self.jobs = recovered.jobs
            self.tasks = [t for j in self.jobs for t in j.tasks]
            for job in self.jobs:
                if job.job_state != JobStatus.finished:
                    job.fail_count = 0
                    job.job_state = JobStatus.unsubmitted
                    job.job_id = ""
                    for task in job.tasks:
                        task.task_state = JobStatus.unsubmitted
            log.info(f"Recovered submission {self.submission_hash}")
        else:
            log.warning("Submission config changed since last run, starting fresh")

    def save_state(self) -> None:
        data = json.dumps(self.serialize(), indent=2, default=str)
        fname = f"{self.submission_hash}.json"
        tmp = fname + ".tmp"
        self.backend.context.write_file(tmp, data)
        self.backend.context.rename_file(tmp, fname)
        log.debug(f"Saved state: {self.submission_hash}.json")

    # --- State management ---

    def _update_job_states(self) -> None:
        for job in self.jobs:
            if job.job_state == JobStatus.finished:
                continue
            job.job_state = self.backend.check_status(job)
            for task in job.tasks:
                if task.task_state != JobStatus.finished:
                    task.task_state = job.job_state  # type: ignore[assignment]
        log.debug(f"Job states: {dict(Counter(j.job_state for j in self.jobs))}")

    def _all_finished(self) -> bool:
        return all(j.job_state == JobStatus.finished for j in self.jobs)

    def _handle_job_states(self, max_retries: int) -> None:
        para_job = self.resources.para_job
        for job in self.jobs:
            if job.job_state == JobStatus.unsubmitted:
                if para_job > 0:
                    active = sum(
                        1
                        for j in self.jobs
                        if j.job_state
                        in (
                            JobStatus.waiting,
                            JobStatus.running,
                            JobStatus.completing,
                        )
                    )
                    if active >= para_job:
                        continue
                self._submit_job(job)
            elif job.job_state == JobStatus.terminated:
                job.fail_count += 1
                self._log_job_failure_info(job)
                if job.fail_count > max_retries:
                    log.info("Downloading backward files before exit")
                    try:
                        self.backend.context.download(self.tasks)
                    except Exception as e:
                        log.warning(f"Failed to download backward files: {e}")
                    raise RuntimeError(
                        f"Job {job.job_hash} failed {job.fail_count} times, "
                        f"exceeding max_retries={max_retries}"
                    )
                log.info(
                    f"Job {job.job_hash} terminated "
                    f"(fail #{job.fail_count}), resubmitting"
                )
                self._submit_job(job)
            elif job.job_state == JobStatus.unknown:
                raise RuntimeError(f"Job {job.job_hash} in unknown state")

    def _submit_job(self, job: Job) -> None:
        job_id = self.backend.do_submit(job)
        job.job_id = job_id
        if job_id:
            job.job_state = JobStatus.waiting
            log.info(f"Submitted job {job.job_hash[:12]} -> job_id={job_id}")
        else:
            job.job_state = JobStatus.unsubmitted
            log.warning(f"Job {job.job_hash[:12]} submission returned empty job_id")

    # --- Main run loop ---

    def run(
        self,
        *,
        check_interval: int = 10,
        max_retries: int = 3,
        clean: bool = True,
    ) -> dict:
        log.info(f"Starting submission {self.submission_hash}")

        # 1. Generate jobs
        if not self.jobs:
            self.generate_jobs()

        # 2. Recovery
        self.try_recover()

        # 3. Check if already finished
        self._update_job_states()
        if self._all_finished():
            log.info("All jobs already finished (recovered)")
            self._download_and_finish(clean)
            return self.serialize()

        # 4. Upload
        log.info("Uploading forward files")
        self.backend.context.upload(self.tasks, self.forward_common_files)
        log.info("Upload complete")

        # 5. Submit
        self._handle_job_states(max_retries)
        self.save_state()

        # 6. Poll
        log.info(
            f"Polling jobs (interval={check_interval}s, max_retries={max_retries})"
        )
        try:
            with tqdm(total=len(self.jobs), desc="Jobs") as pbar:
                while not self._all_finished():
                    time.sleep(check_interval)
                    self._update_job_states()
                    self._handle_job_states(max_retries)
                    self.save_state()
                    pbar.n = sum(
                        1 for j in self.jobs if j.job_state == JobStatus.finished
                    )
                    pbar.refresh()
        except (Exception, KeyboardInterrupt):
            log.exception("Submission interrupted, state saved")
            self.save_state()
            raise

        # 7. Download
        log.info("Downloading backward files")
        self._download_and_finish(clean)
        log.info(f"Submission {self.submission_hash} completed successfully")
        return self.serialize()

    def _log_job_failure_info(self, job: Job) -> None:
        """Fetch and display error diagnostics for a failed job."""
        ctx = self.backend.context
        for task in job.tasks:
            errlog_path = f"{task.task_work_path}/{task.errlog}"
            try:
                if ctx.check_file_exists(errlog_path):
                    content = ctx.read_file(errlog_path).strip()
                    if content:
                        lines = content.splitlines()
                        tail = lines[-30:]
                        prefix = (
                            f"... ({len(lines) - 30} lines omitted)\n"
                            if len(lines) > 30
                            else ""
                        )
                        log.error(
                            f"Task {task.task_work_path} stderr"
                            f" ({task.errlog}):\n{prefix}"
                            + "\n".join(tail)
                        )
            except Exception as e:
                log.debug(
                    f"Could not read errlog for {task.task_work_path}: {e}"
                )
        try:
            exit_info = self.backend.get_exit_info(job)
            if exit_info:
                log.error(f"Job {job.job_hash[:12]} exit info:\n{exit_info}")
        except Exception as e:
            log.debug(
                f"Could not get exit info for {job.job_hash[:12]}: {e}"
            )

    def _download_and_finish(self, clean: bool) -> None:
        self.backend.context.download(self.tasks)
        self.save_state()
        if clean:
            self.backend.context.clean()
