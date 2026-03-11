import os
import shutil
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="minidispatch_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def work_dir(tmp_dir):
    """Create a work directory with some task subdirectories."""
    work = os.path.join(tmp_dir, "work")
    for i in range(3):
        task_dir = os.path.join(work, f"task_{i:03d}")
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "input.txt"), "w") as f:
            f.write(f"task {i}\n")
    return work
