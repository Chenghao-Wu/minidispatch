import os

from minidispatch.backends.shell import Shell
from minidispatch.contexts.local import LocalContext
from minidispatch.job import Job
from minidispatch.models import Resources, Task
from minidispatch.status import JobStatus
from minidispatch.submission import Submission


class TestJobGrouping:
    def test_deterministic_grouping(self):
        """Job generation with seed 42 should be deterministic."""
        tasks = [Task(command=f"echo {i}", task_work_path=f"t{i}") for i in range(6)]
        resources = Resources()
        ctx = LocalContext(local_root="/tmp/l", remote_root="/tmp/r")
        backend = Shell(ctx)
        sub1 = Submission(".", backend, resources, tasks)
        sub1.generate_jobs()

        tasks2 = [Task(command=f"echo {i}", task_work_path=f"t{i}") for i in range(6)]
        resources2 = Resources()
        ctx2 = LocalContext(local_root="/tmp/l", remote_root="/tmp/r")
        backend2 = Shell(ctx2)
        sub2 = Submission(".", backend2, resources2, tasks2)
        sub2.generate_jobs()

        # One job per task
        assert len(sub1.jobs) == len(sub2.jobs) == 6
        for j1, j2 in zip(sub1.jobs, sub2.jobs):
            assert j1 == j2

    def test_one_task_per_job(self):
        """Each task should get its own job."""
        tasks = [Task(command=f"echo {i}", task_work_path=f"t{i}") for i in range(5)]
        resources = Resources()
        ctx = LocalContext(local_root="/tmp/l", remote_root="/tmp/r")
        backend = Shell(ctx)
        sub = Submission(".", backend, resources, tasks)
        sub.generate_jobs()
        assert len(sub.jobs) == 5
        for job in sub.jobs:
            assert len(job.tasks) == 1


class TestSerialization:
    def test_submission_serialize_roundtrip(self):
        tasks = [Task(command="echo", task_work_path="t0")]
        resources = Resources()
        ctx = LocalContext(local_root="/tmp/l", remote_root="/tmp/r")
        backend = Shell(ctx)
        sub = Submission("work", backend, resources, tasks)
        sub.generate_jobs()
        d = sub.serialize()
        assert "work_base" in d
        assert "belonging_jobs" in d

    def test_job_serialize_static(self):
        tasks = [Task(command="echo", task_work_path="t0")]
        resources = Resources()
        job = Job(tasks=tasks, resources=resources)
        job.job_state = JobStatus.running
        job.job_id = "12345"

        static = job.serialize(if_static=True)
        full = job.serialize(if_static=False)
        job_hash = next(iter(static))
        assert "job_state" not in static[job_hash]
        assert full[job_hash]["job_state"] == JobStatus.running

    def test_job_deserialize(self):
        tasks = [Task(command="echo hello", task_work_path="t")]
        resources = Resources()
        job = Job(tasks=tasks, resources=resources)
        job.job_state = JobStatus.finished
        job.job_id = "99"
        job.fail_count = 2

        d = job.serialize()
        job2 = Job.deserialize(d)
        assert job2.job_id == "99"
        assert job2.fail_count == 2
        assert job2.job_state == JobStatus.finished


class TestRecovery:
    def test_recovery_from_json(self, work_dir, tmp_dir):
        """Save state, create new submission, verify recovery works."""
        remote = os.path.join(tmp_dir, "remote")
        tasks = [
            Task(
                command="echo done > output.txt",
                task_work_path="task_000",
                forward_files=["input.txt"],
                backward_files=["output.txt"],
            )
        ]
        resources = Resources()

        # First run
        ctx1 = LocalContext(local_root=work_dir, remote_root=remote)
        backend1 = Shell(ctx1)
        sub1 = Submission(".", backend1, resources, tasks)
        sub1.generate_jobs()
        sub1.jobs[0].job_state = JobStatus.finished
        sub1.jobs[0].job_id = "42"
        sub1.save_state()

        # Second run — should recover
        tasks2 = [
            Task(
                command="echo done > output.txt",
                task_work_path="task_000",
                forward_files=["input.txt"],
                backward_files=["output.txt"],
            )
        ]
        resources2 = Resources()
        ctx2 = LocalContext(local_root=work_dir, remote_root=remote)
        backend2 = Shell(ctx2)
        sub2 = Submission(".", backend2, resources2, tasks2)
        sub2.generate_jobs()
        sub2.try_recover()

        assert sub2.jobs[0].job_id == "42"
        assert sub2.jobs[0].job_state == JobStatus.finished

    def test_corrupted_json_starts_fresh(self, work_dir, tmp_dir):
        """Corrupted recovery JSON should not crash."""
        remote = os.path.join(tmp_dir, "remote")
        tasks = [
            Task(command="echo", task_work_path="task_000", forward_files=["input.txt"])
        ]
        resources = Resources()
        ctx = LocalContext(local_root=work_dir, remote_root=remote)
        backend = Shell(ctx)
        sub = Submission(".", backend, resources, tasks)
        sub.generate_jobs()

        # Write corrupted JSON
        fname = f"{sub.submission_hash}.json"
        os.makedirs(ctx.remote_root, exist_ok=True)
        ctx.write_file(fname, "{ broken json !!!")

        # Should not raise
        sub.try_recover()
        assert sub.jobs[0].job_state is None  # Not recovered
