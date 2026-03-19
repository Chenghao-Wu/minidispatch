from __future__ import annotations

import shlex

from minidispatch._logging import log
from minidispatch.backend import Backend
from minidispatch.job import Job
from minidispatch.status import JobStatus


class Shell(Backend):
    """Local process backend — runs jobs via bash."""

    def gen_script_header(self, job: Job) -> str:
        return "#!/bin/bash -l\n"

    def do_submit(self, job: Job) -> str:
        script_str = self.gen_script(job)
        self.context.write_file(job.script_file_name, script_str)
        script_run_str = self.gen_script_command(job)
        self.context.write_file(f"{job.script_file_name}.run", script_run_str)

        output_name = job.job_hash + ".out"
        remote_root = shlex.quote(self.context.remote_root)  # type: ignore[attr-defined]
        cmd = (
            f"cd {remote_root} && "
            f"{{ nohup bash {job.script_file_name} "
            f"1>>{output_name} 2>>{output_name} & }} && echo $!"
        )
        ret, stdout, stderr = self.context.block_call(cmd)
        if ret != 0:
            raise RuntimeError(f"Submit command failed (rc={ret}): {stderr}")
        job_id = stdout.strip()
        job_id_name = job.job_hash + "_job_id"
        self.context.write_file(job_id_name, job_id)
        log.info(f"Shell: submitted {job.job_hash[:12]} pid={job_id}")
        return job_id

    def check_status(self, job: Job) -> JobStatus:
        if self.check_finish_tag(job):
            return JobStatus.finished
        if job.job_id == "":
            return job.job_state or JobStatus.unsubmitted
        cmd = (
            f"if ps -p {job.job_id} > /dev/null 2>&1 && "
            f"! (ps -o command -p {job.job_id} 2>/dev/null "
            f"| grep defunct >/dev/null) ; "
            f"then echo 1; fi"
        )
        ret, stdout, stderr = self.context.block_call(cmd)
        if ret != 0:
            raise RuntimeError(f"Status check failed (rc={ret}): {stderr}")

        if stdout.strip():
            return JobStatus.running
        log.debug(f"Job {job.job_hash[:12]} process not running, terminated")
        return JobStatus.terminated

    def check_finish_tag(self, job: Job) -> bool:
        return self.context.check_file_exists(job.job_hash + "_job_tag_finished")

    def kill(self, job: Job) -> None:
        if job.job_id:
            log.info(f"Killing job {job.job_hash[:12]} pid={job.job_id}")
            self.context.block_call(f"kill -9 {job.job_id}")

    def get_exit_info(self, job: Job) -> str:
        output_name = job.job_hash + ".out"
        if self.context.check_file_exists(output_name):
            content = self.context.read_file(output_name).strip()
            if content:
                lines = content.splitlines()[-20:]
                return f"script output ({output_name}):\n" + "\n".join(lines)
        return ""

    def gen_script(self, job: Job) -> str:
        """Override to source the .run file instead of inlining commands."""
        from minidispatch.backend import script_template

        return script_template.format(
            script_header=self.gen_script_header(job),
            script_custom_flags=self.gen_script_custom_flags(job),
            script_env=self.gen_script_env(job),
            script_command=f'source "$REMOTE_ROOT"/{job.script_file_name}.run',
            script_end=self.gen_script_end(job),
        )
