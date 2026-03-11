from __future__ import annotations

import json
from hashlib import sha1

from minidispatch.status import JobStatus


class Task:
    """A single command to execute, with files to transfer."""

    def __init__(
        self,
        command: str,
        task_work_path: str,
        forward_files: list[str] | None = None,
        backward_files: list[str] | None = None,
        outlog: str = "log",
        errlog: str = "err",
    ):
        self.command = command
        self.task_work_path = task_work_path
        self.forward_files = forward_files or []
        self.backward_files = backward_files or []
        self.outlog = outlog
        self.errlog = errlog
        self.task_state = JobStatus.unsubmitted

    @property
    def task_hash(self) -> str:
        return sha1(json.dumps(self.serialize()).encode()).hexdigest()

    def serialize(self) -> dict:
        return {
            "command": self.command,
            "task_work_path": self.task_work_path,
            "forward_files": self.forward_files,
            "backward_files": self.backward_files,
            "outlog": self.outlog,
            "errlog": self.errlog,
        }

    @classmethod
    def deserialize(cls, d: dict) -> Task:
        return cls(**d)

    def __repr__(self) -> str:
        return str(self.serialize())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return self.serialize() == other.serialize()


class Resources:
    """Compute resource description for job generation and script rendering."""

    def __init__(
        self,
        group_size: int = 1,
        *,
        number_node: int = 1,
        cpu_per_node: int = 1,
        gpu_per_node: int = 0,
        queue_name: str = "",
        para_deg: int = 1,
        para_job: int = 0,
        module_purge: bool = False,
        module_list: list[str] | None = None,
        module_unload_list: list[str] | None = None,
        source_list: list[str] | None = None,
        prepend_script: list[str] | None = None,
        append_script: list[str] | None = None,
        envs: dict[str, str] | None = None,
        custom_flags: list[str] | None = None,
        if_cuda_multi_devices: bool = False,
        kwargs: dict | None = None,
    ):
        # group_size and para_deg are accepted for backward compat but forced to 1
        self.group_size = 1
        self.number_node = number_node
        self.cpu_per_node = cpu_per_node
        self.gpu_per_node = gpu_per_node
        self.queue_name = queue_name
        self.para_deg = 1
        self.para_job = para_job
        self.module_purge = module_purge
        self.module_list = module_list or []
        self.module_unload_list = module_unload_list or []
        self.source_list = source_list or []
        self.prepend_script = prepend_script or []
        self.append_script = append_script or []
        self.envs = envs or {}
        self.custom_flags = custom_flags or []
        self.if_cuda_multi_devices = if_cuda_multi_devices
        self.kwargs = kwargs or {}

        if self.if_cuda_multi_devices:
            if gpu_per_node < 1:
                raise ValueError(
                    "gpu_per_node must be >= 1 when if_cuda_multi_devices is True"
                )
            if number_node != 1:
                raise ValueError(
                    "number_node must be 1 when if_cuda_multi_devices is True"
                )

    def serialize(self) -> dict:
        return {
            "group_size": self.group_size,
            "number_node": self.number_node,
            "cpu_per_node": self.cpu_per_node,
            "gpu_per_node": self.gpu_per_node,
            "queue_name": self.queue_name,
            "para_deg": self.para_deg,
            "para_job": self.para_job,
            "module_purge": self.module_purge,
            "module_list": self.module_list,
            "module_unload_list": self.module_unload_list,
            "source_list": self.source_list,
            "prepend_script": self.prepend_script,
            "append_script": self.append_script,
            "envs": self.envs,
            "custom_flags": self.custom_flags,
            "if_cuda_multi_devices": self.if_cuda_multi_devices,
            "kwargs": self.kwargs,
        }

    @classmethod
    def deserialize(cls, d: dict) -> Resources:
        return cls(
            group_size=d.get("group_size", 1),
            number_node=d.get("number_node", 1),
            cpu_per_node=d.get("cpu_per_node", 1),
            gpu_per_node=d.get("gpu_per_node", 0),
            queue_name=d.get("queue_name", ""),
            para_deg=d.get("para_deg", 1),
            para_job=d.get("para_job", 0),
            module_purge=d.get("module_purge", False),
            module_list=d.get("module_list", []),
            module_unload_list=d.get("module_unload_list", []),
            source_list=d.get("source_list", []),
            prepend_script=d.get("prepend_script", []),
            append_script=d.get("append_script", []),
            envs=d.get("envs", {}),
            custom_flags=d.get("custom_flags", []),
            if_cuda_multi_devices=d.get("if_cuda_multi_devices", False),
            kwargs=d.get("kwargs", {}),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resources):
            return NotImplemented
        return self.serialize() == other.serialize()

    def __repr__(self) -> str:
        return str(self.serialize())
