"""Tests for SSHContext with paramiko mocking."""

import os
from unittest.mock import MagicMock, patch

import pytest

try:
    import paramiko

    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False

pytestmark = pytest.mark.skipif(not HAS_PARAMIKO, reason="paramiko not installed")


class TestSSHContextUnit:
    def test_registry(self):
        from minidispatch.context import BaseContext

        assert "sshcontext" in BaseContext._registry
        assert "ssh" in BaseContext._registry

    @patch("minidispatch.contexts.ssh.paramiko")
    def test_block_call(self, mock_paramiko):
        from minidispatch.contexts.ssh import SSHContext

        ctx = SSHContext(
            local_root="/tmp/local",
            remote_root="/tmp/remote",
            remote_profile={"hostname": "example.com", "username": "user"},
        )
        ctx.bind_submission("work", "hash123")

        # Mock SSH client
        mock_ssh = MagicMock()
        mock_transport = MagicMock()
        mock_transport.is_active.return_value = True
        mock_ssh.get_transport.return_value = mock_transport

        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"output"
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""

        mock_ssh.exec_command.return_value = (MagicMock(), mock_stdout, mock_stderr)
        ctx._ssh = mock_ssh

        ret, stdout, stderr = ctx.block_call("ls")
        assert ret == 0
        assert stdout == "output"

    def test_serialize(self):
        from minidispatch.contexts.ssh import SSHContext

        ctx = SSHContext(
            local_root="/tmp/local",
            remote_root="/tmp/remote",
            remote_profile={"hostname": "h", "username": "u"},
        )
        d = ctx.serialize()
        assert d["context_type"] == "SSHContext"
        assert d["remote_profile"]["hostname"] == "h"
