from __future__ import annotations

import json
from hashlib import sha1

from minidispatch.models import Resources, Task
from minidispatch.status import JobStatus


class Job:
    """A group of tasks to be submitted as a single batch job."""

    def __init__(self, tasks: list[Task], resources: Resources):
        self.tasks = tasks
        self.resources = resources
        self.job_id: str = ""
        self.job_state: JobStatus | None = None
        self.fail_count: int = 0
        self.script_file_name = self.job_hash + ".sub"

    @property
    def job_hash(self) -> str:
        content = {
            "job_task_list": [t.serialize() for t in self.tasks],
            "resources": self.resources.serialize(),
        }
        return sha1(json.dumps(content).encode()).hexdigest()

    def serialize(self, if_static: bool = False) -> dict:
        content: dict = {
            "job_task_list": [t.serialize() for t in self.tasks],
            "resources": self.resources.serialize(),
        }
        job_hash = sha1(json.dumps(content).encode()).hexdigest()
        if not if_static:
            content["job_state"] = self.job_state
            content["job_id"] = self.job_id
            content["fail_count"] = self.fail_count
        return {job_hash: content}

    @classmethod
    def deserialize(cls, job_dict: dict) -> Job:
        if len(job_dict) != 1:
            raise ValueError(f"Job dict must have exactly 1 key, got {len(job_dict)}")
        job_hash = next(iter(job_dict))
        data = job_dict[job_hash]
        tasks = [Task.deserialize(td) for td in data["job_task_list"]]
        job = cls(tasks=tasks, resources=Resources.deserialize(data["resources"]))
        job.job_state = data.get("job_state")
        job.job_id = data.get("job_id", "")
        job.fail_count = data.get("fail_count", 0)
        for task in job.tasks:
            task.task_state = job.job_state  # type: ignore[assignment]
        return job

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Job):
            return NotImplemented
        return self.serialize(if_static=True) == other.serialize(if_static=True)

    def __repr__(self) -> str:
        return str(self.serialize())
