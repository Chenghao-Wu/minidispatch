from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from pathlib import PurePosixPath

from minidispatch.context import BaseContext
from minidispatch.job import Job
from minidispatch.status import JobStatus

script_template = """\
{script_header}
{script_custom_flags}
{script_env}
{script_command}
{script_end}
"""

script_env_template = """
REMOTE_ROOT=$(cd {remote_root} && pwd -P)
echo 0 > "$REMOTE_ROOT"/{flag_if_job_task_fail}
test $? -ne 0 && exit 1

{module_unload_part}
{module_load_part}
{source_files_part}
{export_envs_part}
{prepend_script_part}
"""

script_command_template = """
cd "$REMOTE_ROOT"
cd {task_work_path}
test $? -ne 0 && exit 1
if [ ! -f {task_tag_finished} ] ;then
  {command_env} ( {command} ) {log_err_part}
  if test $? -eq 0; then touch {task_tag_finished}
  else echo 1 > "$REMOTE_ROOT"/{flag_if_job_task_fail};fi
fi &
"""

script_end_template = """
cd "$REMOTE_ROOT"
test $? -ne 0 && exit 1

wait
FLAG_IF_JOB_TASK_FAIL=$(cat {flag_if_job_task_fail})
if test $FLAG_IF_JOB_TASK_FAIL -eq 0; then touch {job_tag_finished}; else exit 1;fi

{append_script_part}
"""


class Backend(ABC):
    """Abstract batch system backend."""

    _registry: dict[str, type[Backend]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        Backend._registry[cls.__name__.lower()] = cls

    def __init__(self, context: BaseContext) -> None:
        self.context = context

    @classmethod
    def create(cls, name: str, context: BaseContext) -> Backend:
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown backend {name!r}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key](context)

    @classmethod
    def from_dict(cls, d: dict) -> Backend:
        context = BaseContext.create(
            d["context_type"],
            local_root=d["local_root"],
            remote_root=d["remote_root"],
            remote_profile=d.get("remote_profile", {}),
        )
        return cls.create(d["batch_type"], context)

    def serialize(self) -> dict:
        return {
            "batch_type": self.__class__.__name__,
            **self.context.serialize(),
        }

    # --- Script generation ---

    def gen_script(self, job: Job) -> str:
        return script_template.format(
            script_header=self.gen_script_header(job),
            script_custom_flags=self.gen_script_custom_flags(job),
            script_env=self.gen_script_env(job),
            script_command=self.gen_script_command(job),
            script_end=self.gen_script_end(job),
        )

    @abstractmethod
    def gen_script_header(self, job: Job) -> str: ...

    def gen_script_custom_flags(self, job: Job) -> str:
        return "\n".join(job.resources.custom_flags)

    def gen_script_env(self, job: Job) -> str:
        res = job.resources
        module_unload_part = ""
        if res.module_purge:
            module_unload_part += "module purge\n"
        for mod in res.module_unload_list:
            module_unload_part += f"module unload {mod}\n"

        module_load_part = ""
        for mod in res.module_list:
            module_load_part += f"module load {mod}\n"

        source_files_part = ""
        for src in res.source_list:
            source_files_part += f"source {src}\n"

        export_envs_part = ""
        for k, v in res.envs.items():
            if isinstance(v, list):
                for each in v:
                    export_envs_part += f"export {k}={each}\n"
            else:
                export_envs_part += f"export {k}={v}\n"

        prepend_script_part = "\n".join(res.prepend_script)
        flag_if_job_task_fail = job.job_hash + "_flag_if_job_task_fail"

        return script_env_template.format(
            flag_if_job_task_fail=flag_if_job_task_fail,
            remote_root=shlex.quote(self.context.remote_root),  # type: ignore[attr-defined]
            module_unload_part=module_unload_part,
            module_load_part=module_load_part,
            source_files_part=source_files_part,
            export_envs_part=export_envs_part,
            prepend_script_part=prepend_script_part,
        )

    def gen_script_command(self, job: Job) -> str:
        script_command = ""
        resources = job.resources
        command_env = ""
        if resources.if_cuda_multi_devices:
            command_env = "export CUDA_VISIBLE_DEVICES=0;"
        for task in job.tasks:
            task_tag_finished = task.task_hash + "_task_tag_finished"

            log_err_part = ""
            if task.outlog is not None:
                log_err_part += f"1>>{shlex.quote(task.outlog)} "
            if task.errlog is not None:
                log_err_part += f"2>>{shlex.quote(task.errlog)} "

            flag_if_job_task_fail = job.job_hash + "_flag_if_job_task_fail"
            single = script_command_template.format(
                flag_if_job_task_fail=flag_if_job_task_fail,
                command_env=command_env,
                task_work_path=shlex.quote(
                    PurePosixPath(task.task_work_path).as_posix()
                ),
                command=task.command,
                task_tag_finished=task_tag_finished,
                log_err_part=log_err_part,
            )
            script_command += single
        return script_command

    def gen_script_end(self, job: Job) -> str:
        job_tag_finished = job.job_hash + "_job_tag_finished"
        flag_if_job_task_fail = job.job_hash + "_flag_if_job_task_fail"
        append_script_part = "\n".join(job.resources.append_script)

        return script_end_template.format(
            job_tag_finished=job_tag_finished,
            flag_if_job_task_fail=flag_if_job_task_fail,
            append_script_part=append_script_part,
        )

    # --- Abstract batch operations ---

    @abstractmethod
    def do_submit(self, job: Job) -> str: ...

    @abstractmethod
    def check_status(self, job: Job) -> JobStatus: ...

    @abstractmethod
    def check_finish_tag(self, job: Job) -> bool: ...

    def get_exit_info(self, job: Job) -> str:
        """Return human-readable exit info for a failed job. Override in subclass."""
        return ""

    def kill(self, job: Job) -> None:
        """Kill a running job. Override in subclass."""
