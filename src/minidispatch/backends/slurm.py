from __future__ import annotations

import shlex

from minidispatch._logging import log
from minidispatch.backend import Backend
from minidispatch.job import Job
from minidispatch.status import JobStatus

slurm_script_header_template = """\
#!/bin/bash -l
#SBATCH --parsable
{slurm_nodes_line}
{slurm_ntasks_per_node_line}
{slurm_number_gpu_line}
{slurm_partition_line}"""


class Slurm(Backend):
    """Slurm batch system backend."""

    def gen_script_header(self, job: Job) -> str:
        res = job.resources
        header: dict[str, str] = {}
        header["slurm_nodes_line"] = f"#SBATCH --nodes {res.number_node}"
        header["slurm_ntasks_per_node_line"] = (
            f"#SBATCH --ntasks-per-node {res.cpu_per_node}"
        )
        custom_gpu = res.kwargs.get("custom_gpu_line")
        if custom_gpu:
            header["slurm_number_gpu_line"] = custom_gpu
        else:
            header["slurm_number_gpu_line"] = f"#SBATCH --gres=gpu:{res.gpu_per_node}"
        if res.queue_name:
            header["slurm_partition_line"] = f"#SBATCH --partition {res.queue_name}"
        else:
            header["slurm_partition_line"] = ""
        return slurm_script_header_template.format(**header)

    def do_submit(self, job: Job) -> str:
        script_str = self.gen_script(job)
        self.context.write_file(job.script_file_name, script_str)
        script_run_str = self.gen_script_command(job)
        self.context.write_file(f"{job.script_file_name}.run", script_run_str)

        remote_root = shlex.quote(self.context.remote_root)  # type: ignore[attr-defined]
        script = shlex.quote(job.script_file_name)
        cmd = f"cd {remote_root} && sbatch --parsable {script}"
        ret, stdout, stderr = self.context.block_call(cmd)
        if ret != 0:
            raise RuntimeError(f"sbatch failed (rc={ret}): {stderr}")
        # --parsable output: job_id[;cluster_name]
        job_id = stdout.strip().split(";")[0]
        job_id_name = job.job_hash + "_job_id"
        self.context.write_file(job_id_name, job_id)
        log.info(f"Slurm: submitted {job.job_hash[:12]} job_id={job_id}")
        return job_id

    def check_status(self, job: Job) -> JobStatus:
        if self.check_finish_tag(job):
            return JobStatus.finished
        if job.job_id == "":
            return job.job_state or JobStatus.unsubmitted
        cmd = f'squeue -o "%.18i %.2t" -j {job.job_id}'
        ret, stdout, stderr = self.context.block_call(cmd)
        if ret != 0:
            if "Invalid job id specified" in stderr:
                if self.check_finish_tag(job):
                    log.info(f"Job {job.job_hash} {job.job_id} finished")
                    return JobStatus.finished
                log.debug(
                    f"Job {job.job_hash[:12]} not in squeue, "
                    "no finish tag -> terminated"
                )
                return JobStatus.terminated
            raise RuntimeError(f"squeue failed (rc={ret}): {stderr}")
        status_line = stdout.strip().split("\n")[-1]
        parts = status_line.split()
        if len(parts) != 2:
            raise RuntimeError(f"Cannot parse squeue output: {status_line!r}")
        status_word = parts[-1]
        if status_word in ("PD", "CF", "S"):
            return JobStatus.waiting
        if status_word == "R":
            return JobStatus.running
        if status_word == "CG":
            return JobStatus.completing
        if status_word in (
            "C",
            "E",
            "K",
            "BF",
            "CA",
            "CD",
            "F",
            "NF",
            "PR",
            "SE",
            "ST",
            "TO",
        ):
            if self.check_finish_tag(job):
                log.info(f"Job {job.job_hash} {job.job_id} finished")
                return JobStatus.finished
            return JobStatus.terminated
        log.warning(f"Job {job.job_hash[:12]} unknown squeue status: {status_word}")
        return JobStatus.unknown

    def check_finish_tag(self, job: Job) -> bool:
        return self.context.check_file_exists(job.job_hash + "_job_tag_finished")

    def kill(self, job: Job) -> None:
        if job.job_id:
            log.info(f"Cancelling job {job.job_hash[:12]} job_id={job.job_id}")
            self.context.block_call(f"scancel -Q {job.job_id}")

    def get_exit_info(self, job: Job) -> str:
        if not job.job_id:
            return ""
        parts = []
        cmd = (
            f"sacct -j {job.job_id} "
            f"--format=JobID,State,ExitCode,Reason "
            f"--noheader --parsable2"
        )
        ret, stdout, _ = self.context.block_call(cmd)
        if ret == 0 and stdout.strip():
            parts.append(f"sacct: {stdout.strip()}")
        slurm_out = f"slurm-{job.job_id}.out"
        if self.context.check_file_exists(slurm_out):
            content = self.context.read_file(slurm_out).strip()
            if content:
                lines = content.splitlines()[-20:]
                parts.append(f"slurm output ({slurm_out}):\n" + "\n".join(lines))
        return "\n".join(parts)

    def gen_script(self, job: Job) -> str:
        from minidispatch.backend import script_template

        return script_template.format(
            script_header=self.gen_script_header(job),
            script_custom_flags=self.gen_script_custom_flags(job),
            script_env=self.gen_script_env(job),
            script_command=f'source "$REMOTE_ROOT"/{job.script_file_name}.run',
            script_end=self.gen_script_end(job),
        )
