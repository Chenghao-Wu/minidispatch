from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from minidispatch.cli import _load_task_dicts, _validate_config, main


class TestValidateConfig:
    """Tests for _validate_config migration guards."""

    def test_missing_required_keys(self):
        cfg = {"batch_type": "shell"}
        with pytest.raises(ValueError, match="Config missing required keys"):
            _validate_config(cfg)

    def test_nested_machine_error(self):
        cfg = {
            "batch_type": "shell",
            "context_type": "local",
            "local_root": "/tmp/l",
            "remote_root": "/tmp/r",
            "machine": "shell",
        }
        with pytest.raises(ValueError, match="Nested 'machine' section"):
            _validate_config(cfg)

    def test_nested_resources_error(self):
        cfg = {
            "batch_type": "shell",
            "context_type": "local",
            "local_root": "/tmp/l",
            "remote_root": "/tmp/r",
            "resources": "gpu",
        }
        with pytest.raises(ValueError, match="Nested 'resources' section"):
            _validate_config(cfg)

    def test_inline_tasks_in_config_error(self):
        cfg = {
            "batch_type": "shell",
            "context_type": "local",
            "local_root": "/tmp/l",
            "remote_root": "/tmp/r",
            "tasks": [{"command": "echo hi"}],
        }
        with pytest.raises(ValueError, match="Inline 'tasks' in config"):
            _validate_config(cfg)

    def test_tasks_file_in_config_error(self):
        cfg = {
            "batch_type": "shell",
            "context_type": "local",
            "local_root": "/tmp/l",
            "remote_root": "/tmp/r",
            "tasks_file": "tasks.yaml",
        }
        with pytest.raises(ValueError, match="Pass the task file as the second"):
            _validate_config(cfg)

    def test_work_base_in_config_error(self):
        cfg = {
            "batch_type": "shell",
            "context_type": "local",
            "local_root": "/tmp/l",
            "remote_root": "/tmp/r",
            "work_base": "my_work",
        }
        with pytest.raises(ValueError, match="Use --work-base flag"):
            _validate_config(cfg)

    def test_forward_common_files_in_config_error(self):
        cfg = {
            "batch_type": "shell",
            "context_type": "local",
            "local_root": "/tmp/l",
            "remote_root": "/tmp/r",
            "forward_common_files": ["input.dat"],
        }
        with pytest.raises(ValueError, match="Use --common-files flag"):
            _validate_config(cfg)

    def test_valid_flat_config(self):
        cfg = {
            "batch_type": "shell",
            "context_type": "local",
            "local_root": "/tmp/local",
            "remote_root": "/tmp/remote",
            "para_job": 5,
        }
        _validate_config(cfg)  # should not raise


class TestLoadTaskDicts:
    """Tests for _load_task_dicts."""

    def test_single_task_dict(self, tmp_path):
        task_file = tmp_path / "task.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "command": "echo hello",
                    "task_work_path": "t0",
                }
            )
        )
        tasks, work_base, fwd = _load_task_dicts(task_file)
        assert len(tasks) == 1
        assert tasks[0]["command"] == "echo hello"
        assert work_base == "."
        assert fwd == []

    def test_multiple_tasks_list(self, tmp_path):
        task_data = [
            {"command": "echo 0", "task_work_path": "t0"},
            {"command": "echo 1", "task_work_path": "t1"},
        ]
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(yaml.dump(task_data))
        tasks, work_base, fwd = _load_task_dicts(task_file)
        assert len(tasks) == 2
        assert tasks[1]["command"] == "echo 1"
        assert work_base == "."
        assert fwd == []

    def test_enriched_multi_task(self, tmp_path):
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "work_base": "my_work",
                    "forward_common_files": ["input.dat"],
                    "tasks": [
                        {"command": "echo 0", "task_work_path": "t0"},
                        {"command": "echo 1", "task_work_path": "t1"},
                    ],
                }
            )
        )
        tasks, work_base, fwd = _load_task_dicts(task_file)
        assert len(tasks) == 2
        assert work_base == "my_work"
        assert fwd == ["input.dat"]

    def test_enriched_defaults(self, tmp_path):
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "tasks": [
                        {"command": "echo 0", "task_work_path": "t0"},
                    ],
                }
            )
        )
        tasks, work_base, fwd = _load_task_dicts(task_file)
        assert len(tasks) == 1
        assert work_base == "."
        assert fwd == []

    def test_tasks_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Tasks file not found"):
            _load_task_dicts(tmp_path / "missing.yaml")

    def test_invalid_task_format_string(self, tmp_path):
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text('"just a string"')
        with pytest.raises(ValueError, match="must be a YAML dict.*or list"):
            _load_task_dicts(task_file)

    def test_single_task_missing_command(self, tmp_path):
        task_file = tmp_path / "task.yaml"
        task_file.write_text(yaml.dump({"task_work_path": "t0"}))
        with pytest.raises(ValueError, match="must have 'command' or 'tasks' key"):
            _load_task_dicts(task_file)


class TestCLIOverrides:
    """Tests for CLI flag vs task file precedence."""

    def test_work_base_from_task_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "batch_type": "shell",
                    "context_type": "local",
                    "local_root": str(tmp_path / "local"),
                    "remote_root": str(tmp_path / "remote"),
                }
            )
        )
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "work_base": "from_file",
                    "tasks": [{"command": "echo hi", "task_work_path": "t0"}],
                }
            )
        )
        with patch("minidispatch.cli.Submission") as mock_sub:
            mock_sub.return_value.run.return_value = None
            main([str(config_file), str(task_file)])
            assert mock_sub.call_args.kwargs["work_base"] == "from_file"

    def test_work_base_cli_overrides_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "batch_type": "shell",
                    "context_type": "local",
                    "local_root": str(tmp_path / "local"),
                    "remote_root": str(tmp_path / "remote"),
                }
            )
        )
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "work_base": "from_file",
                    "tasks": [{"command": "echo hi", "task_work_path": "t0"}],
                }
            )
        )
        with patch("minidispatch.cli.Submission") as mock_sub:
            mock_sub.return_value.run.return_value = None
            main([str(config_file), str(task_file), "--work-base", "from_cli"])
            assert mock_sub.call_args.kwargs["work_base"] == "from_cli"

    def test_common_files_from_task_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "batch_type": "shell",
                    "context_type": "local",
                    "local_root": str(tmp_path / "local"),
                    "remote_root": str(tmp_path / "remote"),
                }
            )
        )
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "forward_common_files": ["input.dat"],
                    "tasks": [{"command": "echo hi", "task_work_path": "t0"}],
                }
            )
        )
        with patch("minidispatch.cli.Submission") as mock_sub:
            mock_sub.return_value.run.return_value = None
            main([str(config_file), str(task_file)])
            assert mock_sub.call_args.kwargs["forward_common_files"] == ["input.dat"]

    def test_common_files_cli_overrides_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "batch_type": "shell",
                    "context_type": "local",
                    "local_root": str(tmp_path / "local"),
                    "remote_root": str(tmp_path / "remote"),
                }
            )
        )
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            yaml.dump(
                {
                    "forward_common_files": ["input.dat"],
                    "tasks": [{"command": "echo hi", "task_work_path": "t0"}],
                }
            )
        )
        with patch("minidispatch.cli.Submission") as mock_sub:
            mock_sub.return_value.run.return_value = None
            main(
                [
                    str(config_file),
                    str(task_file),
                    "--common-files",
                    "a.dat",
                    "b.dat",
                ]
            )
            assert mock_sub.call_args.kwargs["forward_common_files"] == [
                "a.dat",
                "b.dat",
            ]


class TestMainE2E:
    """End-to-end test for main()."""

    def test_main_e2e(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "batch_type": "shell",
                    "context_type": "local",
                    "local_root": str(tmp_path / "local"),
                    "remote_root": str(tmp_path / "remote"),
                }
            )
        )
        task_file = tmp_path / "tasks.yaml"
        task_file.write_text(
            yaml.dump(
                [
                    {
                        "command": "echo hello",
                        "task_work_path": "t0",
                        "backward_files": ["log"],
                    },
                ]
            )
        )
        with patch("minidispatch.cli.Submission") as mock_sub:
            mock_sub.return_value.run.return_value = None
            main([str(config_file), str(task_file)])
            mock_sub.assert_called_once()
            mock_sub.return_value.run.assert_called_once_with(
                check_interval=10,
                max_retries=3,
                clean=True,
            )
