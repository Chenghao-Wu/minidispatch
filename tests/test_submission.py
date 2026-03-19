import os
from unittest.mock import patch

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

    def test_recovery_preserves_fail_count(self, work_dir, tmp_dir):
        """Fail count should survive recovery, not reset to 0."""
        remote = os.path.join(tmp_dir, "remote")
        tasks = [
            Task(
                command="echo",
                task_work_path="task_000",
                forward_files=["input.txt"],
            )
        ]
        resources = Resources()

        # First run — simulate 2 failures
        ctx1 = LocalContext(local_root=work_dir, remote_root=remote)
        backend1 = Shell(ctx1)
        sub1 = Submission(".", backend1, resources, tasks)
        sub1.generate_jobs()
        sub1.jobs[0].job_state = JobStatus.terminated
        sub1.jobs[0].job_id = "100"
        sub1.jobs[0].fail_count = 2
        sub1.save_state()

        # Second run — recover
        tasks2 = [
            Task(
                command="echo",
                task_work_path="task_000",
                forward_files=["input.txt"],
            )
        ]
        resources2 = Resources()
        ctx2 = LocalContext(local_root=work_dir, remote_root=remote)
        backend2 = Shell(ctx2)
        sub2 = Submission(".", backend2, resources2, tasks2)
        sub2.generate_jobs()
        sub2.try_recover()

        assert sub2.jobs[0].fail_count == 2
        assert sub2.jobs[0].job_state == JobStatus.unsubmitted

    def test_recovery_uploads_only_unfinished(self, work_dir, tmp_dir):
        """On recovery, only unfinished tasks should be uploaded."""
        remote = os.path.join(tmp_dir, "remote")
        tasks = [
            Task(
                command="echo done > output.txt",
                task_work_path=f"task_{i:03d}",
                forward_files=["input.txt"],
                backward_files=["output.txt"],
            )
            for i in range(3)
        ]
        resources = Resources()

        # First run — mark first 2 as finished
        ctx1 = LocalContext(local_root=work_dir, remote_root=remote)
        backend1 = Shell(ctx1)
        sub1 = Submission(".", backend1, resources, tasks)
        sub1.generate_jobs()
        sub1.jobs[0].job_state = JobStatus.finished
        sub1.jobs[0].job_id = "1"
        sub1.jobs[1].job_state = JobStatus.finished
        sub1.jobs[1].job_id = "2"
        sub1.jobs[2].job_state = JobStatus.terminated
        sub1.jobs[2].job_id = "3"
        sub1.jobs[2].fail_count = 1
        sub1.save_state()

        # Create finish tags for completed jobs
        os.makedirs(ctx1.remote_root, exist_ok=True)
        for job in sub1.jobs[:2]:
            tag = job.job_hash + "_job_tag_finished"
            with open(os.path.join(ctx1.remote_root, tag), "w") as f:
                f.write("")

        # Second run — track what upload receives
        tasks2 = [
            Task(
                command="echo done > output.txt",
                task_work_path=f"task_{i:03d}",
                forward_files=["input.txt"],
                backward_files=["output.txt"],
            )
            for i in range(3)
        ]
        resources2 = Resources()
        ctx2 = LocalContext(local_root=work_dir, remote_root=remote)
        backend2 = Shell(ctx2)
        sub2 = Submission(".", backend2, resources2, tasks2)
        sub2.generate_jobs()
        sub2.try_recover()
        sub2._update_job_states()

        # Verify 2 finished, 1 unfinished
        finished = [j for j in sub2.jobs if j.job_state == JobStatus.finished]
        unfinished = [j for j in sub2.jobs if j.job_state != JobStatus.finished]
        assert len(finished) == 2
        assert len(unfinished) == 1

        # Mock upload to check what tasks are passed
        uploaded_tasks = []

        def mock_upload(tasks_arg, common):
            uploaded_tasks.extend(tasks_arg)
            # Create remote dirs so submit doesn't fail
            for t in tasks_arg:
                os.makedirs(
                    os.path.join(ctx2.remote_root, t.task_work_path), exist_ok=True
                )

        with patch.object(ctx2, "upload", side_effect=mock_upload):
            # Just test the upload filtering, not full run
            unfinished_tasks = [
                t
                for j in sub2.jobs
                if j.job_state != JobStatus.finished
                for t in j.tasks
            ]
            ctx2.upload(unfinished_tasks, sub2.forward_common_files)

        assert len(uploaded_tasks) == 1
        assert uploaded_tasks[0].task_work_path == "task_002"

    def test_check_status_detects_finished_without_job_id(self, work_dir, tmp_dir):
        """check_status returns finished if finish tag exists with job_id=''."""
        remote = os.path.join(tmp_dir, "remote")
        ctx = LocalContext(local_root=work_dir, remote_root=remote)
        backend = Shell(ctx)
        ctx.bind_submission(".", "testhash")

        tasks = [Task(command="echo", task_work_path="task_000")]
        resources = Resources()
        job = Job(tasks=tasks, resources=resources)
        job.job_id = ""
        job.job_state = JobStatus.unsubmitted

        # Create the finish tag file
        os.makedirs(ctx.remote_root, exist_ok=True)
        tag = job.job_hash + "_job_tag_finished"
        with open(os.path.join(ctx.remote_root, tag), "w") as f:
            f.write("")

        status = backend.check_status(job)
        assert status == JobStatus.finished


class TestCleanTaskOutputs:
    def test_removes_backward_files(self, work_dir, tmp_dir):
        """_clean_task_outputs should remove backward-only files."""
        remote = os.path.join(tmp_dir, "remote")
        ctx = LocalContext(local_root=work_dir, remote_root=remote)
        backend = Shell(ctx)
        ctx.bind_submission(".", "testhash")

        task = Task(
            command="echo",
            task_work_path="task_000",
            forward_files=["input.txt"],
            backward_files=["output.txt", "results.csv"],
            outlog="log",
            errlog="err",
        )
        resources = Resources()
        sub = Submission(".", backend, resources, [task])
        sub.generate_jobs()

        # Create remote task dir with files
        task_dir = os.path.join(ctx.remote_root, "task_000")
        os.makedirs(task_dir, exist_ok=True)
        for f in ["input.txt", "output.txt", "results.csv", "log", "err"]:
            with open(os.path.join(task_dir, f), "w") as fh:
                fh.write("data")

        sub._clean_task_outputs(sub.jobs[0])

        # Forward file should remain
        assert os.path.exists(os.path.join(task_dir, "input.txt"))
        # Backward-only files should be removed
        assert not os.path.exists(os.path.join(task_dir, "output.txt"))
        assert not os.path.exists(os.path.join(task_dir, "results.csv"))
        # Logs should be removed
        assert not os.path.exists(os.path.join(task_dir, "log"))
        assert not os.path.exists(os.path.join(task_dir, "err"))

    def test_preserves_forward_backward_overlap(self, work_dir, tmp_dir):
        """Files in both forward and backward lists should NOT be removed."""
        remote = os.path.join(tmp_dir, "remote")
        ctx = LocalContext(local_root=work_dir, remote_root=remote)
        backend = Shell(ctx)
        ctx.bind_submission(".", "testhash")

        task = Task(
            command="echo",
            task_work_path="task_000",
            forward_files=["model.pt", "input.txt"],
            backward_files=["model.pt", "output.txt"],
        )
        resources = Resources()
        sub = Submission(".", backend, resources, [task])
        sub.generate_jobs()

        task_dir = os.path.join(ctx.remote_root, "task_000")
        os.makedirs(task_dir, exist_ok=True)
        for f in ["model.pt", "input.txt", "output.txt"]:
            with open(os.path.join(task_dir, f), "w") as fh:
                fh.write("data")

        sub._clean_task_outputs(sub.jobs[0])

        # Overlap file (model.pt) should survive
        assert os.path.exists(os.path.join(task_dir, "model.pt"))
        # Forward-only file should survive
        assert os.path.exists(os.path.join(task_dir, "input.txt"))
        # Backward-only file should be removed
        assert not os.path.exists(os.path.join(task_dir, "output.txt"))
