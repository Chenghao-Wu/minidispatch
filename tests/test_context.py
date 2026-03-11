import os

from minidispatch.context import BaseContext
from minidispatch.contexts.local import LocalContext


class TestLocalContext:
    def test_registry(self):
        assert "localcontext" in BaseContext._registry
        assert "local" in BaseContext._registry

    def test_create(self, tmp_dir):
        ctx = BaseContext.create(
            "local",
            local_root=tmp_dir,
            remote_root=os.path.join(tmp_dir, "remote"),
        )
        assert isinstance(ctx, LocalContext)

    def test_write_read_file(self, tmp_dir):
        remote = os.path.join(tmp_dir, "remote")
        ctx = LocalContext(local_root=tmp_dir, remote_root=remote)
        ctx.bind_submission("work", "hash123")
        os.makedirs(ctx.remote_root, exist_ok=True)
        ctx.write_file("test.txt", "hello world")
        assert ctx.read_file("test.txt") == "hello world"

    def test_check_file_exists(self, tmp_dir):
        remote = os.path.join(tmp_dir, "remote")
        ctx = LocalContext(local_root=tmp_dir, remote_root=remote)
        ctx.bind_submission("work", "hash123")
        os.makedirs(ctx.remote_root, exist_ok=True)
        assert not ctx.check_file_exists("nonexistent.txt")
        ctx.write_file("exists.txt", "data")
        assert ctx.check_file_exists("exists.txt")

    def test_block_call(self, tmp_dir):
        remote = os.path.join(tmp_dir, "remote")
        os.makedirs(remote)
        ctx = LocalContext(local_root=tmp_dir, remote_root=remote)
        # Don't bind - use the remote_root directly
        ret, stdout, stderr = ctx.block_call("echo hello")
        assert ret == 0
        assert stdout.strip() == "hello"

    def test_rename_file(self, tmp_dir):
        remote = os.path.join(tmp_dir, "remote")
        ctx = LocalContext(local_root=tmp_dir, remote_root=remote)
        ctx.bind_submission("work", "hash456")
        os.makedirs(ctx.remote_root, exist_ok=True)
        ctx.write_file("a.txt", "data")
        ctx.rename_file("a.txt", "b.txt")
        assert not ctx.check_file_exists("a.txt")
        assert ctx.read_file("b.txt") == "data"

    def test_upload_download(self, work_dir, tmp_dir):
        from minidispatch.models import Task

        remote = os.path.join(tmp_dir, "remote")
        ctx = LocalContext(local_root=work_dir, remote_root=remote)
        ctx.bind_submission(".", "test_hash")

        tasks = [
            Task(
                command="cat input.txt > output.txt",
                task_work_path=f"task_{i:03d}",
                forward_files=["input.txt"],
                backward_files=["output.txt"],
            )
            for i in range(3)
        ]

        ctx.upload(tasks, [])

        # Check files were copied
        for i in range(3):
            assert os.path.exists(
                os.path.join(ctx.remote_root, f"task_{i:03d}", "input.txt")
            )

        # Create output files in remote
        for i in range(3):
            out_path = os.path.join(ctx.remote_root, f"task_{i:03d}", "output.txt")
            with open(out_path, "w") as f:
                f.write(f"output {i}")

        ctx.download(tasks)

        # Check files were downloaded
        for i in range(3):
            local_out = os.path.join(ctx.local_root, f"task_{i:03d}", "output.txt")
            assert os.path.exists(local_out)
            with open(local_out) as f:
                assert f.read() == f"output {i}"

    def test_serialize(self, tmp_dir):
        ctx = LocalContext(
            local_root="/tmp/local",
            remote_root="/tmp/remote",
            remote_profile={"key": "val"},
        )
        d = ctx.serialize()
        assert d["context_type"] == "LocalContext"
        assert d["local_root"] == "/tmp/local"
        assert d["remote_root"] == "/tmp/remote"
        assert d["remote_profile"] == {"key": "val"}
