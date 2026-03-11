from __future__ import annotations

import os
import shutil
import subprocess
from glob import glob

from tqdm import tqdm

from minidispatch._logging import log
from minidispatch.context import BaseContext


class LocalContext(BaseContext):
    """Execute jobs on the local filesystem."""

    def __init__(
        self,
        local_root: str,
        remote_root: str,
        remote_profile: dict | None = None,
    ):
        self.init_local_root = local_root
        self.init_remote_root = remote_root
        self.remote_profile = remote_profile or {}
        self.local_root = os.path.abspath(local_root)
        self.remote_root = os.path.abspath(remote_root)

    def bind_submission(self, work_base: str, submission_hash: str) -> None:
        self.local_root = os.path.join(os.path.abspath(self.init_local_root), work_base)
        self.remote_root = os.path.join(
            os.path.abspath(self.init_remote_root), submission_hash
        )

    def upload(self, tasks: list, forward_common_files: list[str]) -> None:
        log.info(f"Local upload: {len(tasks)} tasks to {self.remote_root}")
        os.makedirs(self.remote_root, exist_ok=True)
        for task in tqdm(tasks, desc="Uploading", leave=False):
            local_job = os.path.join(self.local_root, task.task_work_path)
            remote_job = os.path.join(self.remote_root, task.task_work_path)
            os.makedirs(remote_job, exist_ok=True)

            file_list: list[str] = []
            for pattern in task.forward_files:
                abs_matches = glob(os.path.join(local_job, pattern))
                if not abs_matches:
                    raise FileNotFoundError(
                        f"Cannot find upload file {os.path.join(local_job, pattern)}"
                    )
                file_list.extend(
                    os.path.relpath(f, start=local_job) for f in abs_matches
                )

            for fname in file_list:
                src = os.path.join(local_job, fname)
                dst = os.path.join(remote_job, fname)
                self._copy(src, dst)

        # Forward common files
        for pattern in forward_common_files:
            abs_matches = glob(os.path.join(self.local_root, pattern))
            if not abs_matches:
                raise FileNotFoundError(
                    f"Cannot find upload file {os.path.join(self.local_root, pattern)}"
                )
            for f in abs_matches:
                rel = os.path.relpath(f, start=self.local_root)
                self._copy(f, os.path.join(self.remote_root, rel))

    def download(self, tasks: list) -> None:
        log.info(f"Local download: {len(tasks)} tasks from {self.remote_root}")
        for task in tqdm(tasks, desc="Downloading", leave=False):
            local_job = os.path.join(self.local_root, task.task_work_path)
            remote_job = os.path.join(self.remote_root, task.task_work_path)
            for pattern in task.backward_files:
                abs_matches = glob(os.path.join(remote_job, pattern))
                for src in abs_matches:
                    rel = os.path.relpath(src, start=remote_job)
                    dst = os.path.join(local_job, rel)
                    if os.path.realpath(src) != os.path.realpath(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)

    def write_file(self, fname: str, content: str) -> None:
        fpath = os.path.join(self.remote_root, fname)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w") as f:
            f.write(content)

    def read_file(self, fname: str) -> str:
        fpath = os.path.join(self.remote_root, fname)
        with open(fpath) as f:
            return f.read()

    def check_file_exists(self, fname: str) -> bool:
        return os.path.exists(os.path.join(self.remote_root, fname))

    def block_call(self, cmd: str) -> tuple[int, str, str]:
        log.debug(f"Local exec: {cmd}")
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=self.remote_root,
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def clean(self) -> None:
        if os.path.exists(self.remote_root):
            shutil.rmtree(self.remote_root)
            log.info(f"Cleaned remote root: {self.remote_root}")

    def rename_file(self, src: str, dst: str) -> None:
        src_path = os.path.join(self.remote_root, src)
        dst_path = os.path.join(self.remote_root, dst)
        os.rename(src_path, dst_path)

    @staticmethod
    def _copy(src: str, dst: str) -> None:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
