"""minidispatch: simplified job dispatch with recovery and CUDA multi-device."""

# Trigger auto-registration of backends and contexts
import minidispatch.backends  # noqa: F401
import minidispatch.contexts  # noqa: F401
from minidispatch.backend import Backend
from minidispatch.context import BaseContext
from minidispatch.job import Job
from minidispatch.models import Resources, Task
from minidispatch.status import JobStatus
from minidispatch.submission import Submission

__all__ = [
    "Backend",
    "BaseContext",
    "Job",
    "JobStatus",
    "Resources",
    "Submission",
    "Task",
]
