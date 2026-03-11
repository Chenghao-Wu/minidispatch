from __future__ import annotations

import fnmatch
import os
import stat
import time

import paramiko
from tqdm import tqdm

from minidispatch._logging import log
from minidispatch.context import BaseContext


class SSHContext(BaseContext):
    """Execute jobs on a remote machine via SSH/SFTP."""

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
        self.remote_root = remote_root

        self._ssh: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def bind_submission(self, work_base: str, submission_hash: str) -> None:
        self.local_root = os.path.join(os.path.abspath(self.init_local_root), work_base)
        self.remote_root = f"{self.init_remote_root}/{submission_hash}"

    # --- SSH connection management ---

    def _connect(self) -> paramiko.SSHClient:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        profile = self.remote_profile
        connect_kwargs: dict = {}
        for key in (
            "hostname",
            "username",
            "password",
            "port",
            "key_filename",
            "passphrase",
            "timeout",
        ):
            if key in profile:
                connect_kwargs[key] = profile[key]
        connect_kwargs.setdefault("port", 22)
        log.info(
            f"SSH connecting to "
            f"{connect_kwargs.get('hostname', '?')}:{connect_kwargs.get('port', 22)}"
        )
        ssh.connect(**connect_kwargs)
        return ssh

    @property
    def ssh(self) -> paramiko.SSHClient:
        transport = self._ssh.get_transport() if self._ssh else None
        if transport is None or not transport.is_active():
            self._sftp = None
            self._ssh = self._ensure_alive()
        return self._ssh

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            self._sftp = self.ssh.open_sftp()
        return self._sftp

    def _ensure_alive(self) -> paramiko.SSHClient:
        for attempt in range(10):
            try:
                client = self._connect()
                if attempt > 0:
                    log.info(
                        f"SSH connection established after {attempt + 1} attempt(s)"
                    )
                return client
            except Exception as e:
                log.warning(f"SSH connect attempt {attempt + 1} failed: {e}")
                if attempt < 9:
                    time.sleep(10)
        raise RuntimeError("Failed to connect via SSH after 10 attempts")

    def _remote_makedirs(self, path: str) -> None:
        """Recursively create remote directories."""
        try:
            self.sftp.stat(path)
            return
        except FileNotFoundError:
            pass
        parent = os.path.dirname(path)
        if parent and parent != path:
            self._remote_makedirs(parent)
        try:
            self.sftp.mkdir(path)
        except IOError:
            pass  # May already exist due to race

    # --- Context interface ---

    def upload(self, tasks: list, forward_common_files: list[str]) -> None:
        log.info(f"SSH upload: {len(tasks)} tasks to {self.remote_root}")
        self._remote_makedirs(self.remote_root)
        for task in tqdm(tasks, desc="Uploading", leave=False):
            local_job = os.path.join(self.local_root, task.task_work_path)
            remote_job = f"{self.remote_root}/{task.task_work_path}"
            self._remote_makedirs(remote_job)

            for pattern in task.forward_files:
                # Simple file upload (no glob on remote)
                local_path = os.path.join(local_job, pattern)
                remote_path = f"{remote_job}/{pattern}"
                if os.path.isdir(local_path):
                    self._put_dir(local_path, remote_path)
                elif os.path.isfile(local_path):
                    self._remote_makedirs(os.path.dirname(remote_path))
                    self.sftp.put(local_path, remote_path)
                else:
                    raise FileNotFoundError(f"Cannot find upload file {local_path}")

        # Forward common files
        for pattern in forward_common_files:
            local_path = os.path.join(self.local_root, pattern)
            remote_path = f"{self.remote_root}/{pattern}"
            if os.path.isdir(local_path):
                self._put_dir(local_path, remote_path)
            elif os.path.isfile(local_path):
                self._remote_makedirs(os.path.dirname(remote_path))
                self.sftp.put(local_path, remote_path)
            else:
                raise FileNotFoundError(f"Cannot find upload file {local_path}")

    def download(self, tasks: list) -> None:
        log.info(f"SSH download: {len(tasks)} tasks from {self.remote_root}")
        for task in tqdm(tasks, desc="Downloading", leave=False):
            local_job = os.path.join(self.local_root, task.task_work_path)
            remote_job = f"{self.remote_root}/{task.task_work_path}"
            os.makedirs(local_job, exist_ok=True)
            for pattern in task.backward_files:
                matches = self._remote_glob(remote_job, pattern)
                for remote_path in matches:
                    rel = remote_path[len(remote_job) + 1 :]
                    local_path = os.path.join(local_job, rel)
                    try:
                        attr = self.sftp.stat(remote_path)
                        if stat.S_ISDIR(attr.st_mode):  # type: ignore[arg-type]
                            self._get_dir(remote_path, local_path)
                        else:
                            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                            self.sftp.get(remote_path, local_path)
                    except FileNotFoundError:
                        log.warning(f"Backward file not found: {remote_path}")

    def write_file(self, fname: str, content: str) -> None:
        fpath = f"{self.remote_root}/{fname}"
        self._remote_makedirs(os.path.dirname(fpath))
        with self.sftp.open(fpath, "w") as f:
            f.write(content)

    def read_file(self, fname: str) -> str:
        fpath = f"{self.remote_root}/{fname}"
        with self.sftp.open(fpath, "r") as f:
            data = f.read()
            return data.decode() if isinstance(data, bytes) else data

    def check_file_exists(self, fname: str) -> bool:
        fpath = f"{self.remote_root}/{fname}"
        try:
            self.sftp.stat(fpath)
            return True
        except FileNotFoundError:
            return False

    def block_call(self, cmd: str) -> tuple[int, str, str]:
        log.debug(f"SSH exec: {cmd}")
        _, stdout, stderr = self.ssh.exec_command(f"cd {self.remote_root} && {cmd}")
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, stdout.read().decode(), stderr.read().decode()

    def clean(self) -> None:
        self.block_call(f"rm -rf {self.remote_root}")
        log.info(f"Cleaned remote root: {self.remote_root}")

    def rename_file(self, src: str, dst: str) -> None:
        src_path = f"{self.remote_root}/{src}"
        dst_path = f"{self.remote_root}/{dst}"
        # Remove dst first in case it exists (SFTP rename doesn't overwrite)
        try:
            self.sftp.remove(dst_path)
        except FileNotFoundError:
            pass
        self.sftp.rename(src_path, dst_path)

    # --- Glob expansion ---

    def _remote_glob(self, base_dir: str, pattern: str) -> list[str]:
        """Expand a glob pattern relative to base_dir on the remote via SFTP."""
        if not any(c in pattern for c in ("*", "?", "[")):
            return [f"{base_dir}/{pattern}"]
        parts = pattern.split("/")
        return self._remote_glob_walk(base_dir, parts)

    def _remote_glob_walk(self, current_dir: str, parts: list[str]) -> list[str]:
        if not parts:
            return [current_dir]
        head, rest = parts[0], parts[1:]
        if not any(c in head for c in ("*", "?", "[")):
            return self._remote_glob_walk(f"{current_dir}/{head}", rest)
        try:
            entries = self.sftp.listdir_attr(current_dir)
        except (FileNotFoundError, IOError):
            return []
        results = []
        for attr in entries:
            name = attr.filename
            if name.startswith(".") and not head.startswith("."):
                continue
            if fnmatch.fnmatch(name, head):
                child = f"{current_dir}/{name}"
                if rest:
                    if stat.S_ISDIR(attr.st_mode):  # type: ignore[arg-type]
                        results.extend(self._remote_glob_walk(child, rest))
                else:
                    results.append(child)
        return results

    # --- Helpers ---

    def _put_dir(self, local_dir: str, remote_dir: str) -> None:
        self._remote_makedirs(remote_dir)
        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = f"{remote_dir}/{item}"
            if os.path.isdir(local_path):
                self._put_dir(local_path, remote_path)
            else:
                self.sftp.put(local_path, remote_path)

    def _get_dir(self, remote_dir: str, local_dir: str) -> None:
        os.makedirs(local_dir, exist_ok=True)
        for attr in self.sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{attr.filename}"
            local_path = os.path.join(local_dir, attr.filename)
            if stat.S_ISDIR(attr.st_mode):  # type: ignore[arg-type]
                self._get_dir(remote_path, local_path)
            else:
                self.sftp.get(remote_path, local_path)
