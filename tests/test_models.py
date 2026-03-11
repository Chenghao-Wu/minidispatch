from minidispatch.models import Resources, Task
from minidispatch.status import JobStatus


class TestTask:
    def test_serialize_roundtrip(self):
        t = Task(
            command="echo hello",
            task_work_path="task_000",
            forward_files=["input.txt"],
            backward_files=["output.txt"],
        )
        d = t.serialize()
        t2 = Task.deserialize(d)
        assert t == t2
        assert t.task_hash == t2.task_hash

    def test_task_hash_deterministic(self):
        t1 = Task(command="echo a", task_work_path="p1")
        t2 = Task(command="echo a", task_work_path="p1")
        assert t1.task_hash == t2.task_hash

    def test_task_hash_differs(self):
        t1 = Task(command="echo a", task_work_path="p1")
        t2 = Task(command="echo b", task_work_path="p1")
        assert t1.task_hash != t2.task_hash

    def test_initial_state(self):
        t = Task(command="echo", task_work_path="p")
        assert t.task_state == JobStatus.unsubmitted

    def test_serialize_excludes_runtime(self):
        t = Task(command="echo", task_work_path="p")
        t.task_state = JobStatus.running
        d = t.serialize()
        assert "task_state" not in d


class TestResources:
    def test_serialize_roundtrip(self):
        r = Resources(
            gpu_per_node=4,
            para_job=5,
            module_list=["cuda/11.0"],
            if_cuda_multi_devices=True,
        )
        d = r.serialize()
        r2 = Resources.deserialize(d)
        assert r == r2

    def test_cuda_validation(self):
        import pytest

        with pytest.raises(ValueError, match="gpu_per_node"):
            Resources(gpu_per_node=0, if_cuda_multi_devices=True)

        with pytest.raises(ValueError, match="number_node"):
            Resources(
                gpu_per_node=2,
                number_node=2,
                if_cuda_multi_devices=True,
            )

    def test_defaults(self):
        r = Resources()
        assert r.number_node == 1
        assert r.cpu_per_node == 1
        assert r.gpu_per_node == 0
        assert r.para_deg == 1
        assert r.para_job == 0
        assert r.module_list == []
        assert r.envs == {}

    def test_group_size_forced_to_1(self):
        r = Resources(group_size=5)
        assert r.group_size == 1

    def test_para_deg_forced_to_1(self):
        r = Resources(para_deg=4)
        assert r.para_deg == 1

    def test_para_job(self):
        r = Resources(para_job=10)
        assert r.para_job == 10
        d = r.serialize()
        assert d["para_job"] == 10
        r2 = Resources.deserialize(d)
        assert r2.para_job == 10
