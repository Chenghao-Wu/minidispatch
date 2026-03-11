from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from minidispatch.backend import Backend
from minidispatch.models import Resources, Task
from minidispatch.submission import Submission


def _validate_config(cfg: dict) -> None:
    """Check for required keys and legacy config formats."""
    required = {"batch_type", "context_type", "local_root", "remote_root"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(
            f"Config missing required keys: {', '.join(sorted(missing))}. "
            f"Required: {', '.join(sorted(required))}"
        )

    if "machine" in cfg:
        raise ValueError(
            "Nested 'machine' section is no longer supported. "
            "Move machine keys to the top level of the config file."
        )
    if "resources" in cfg:
        raise ValueError(
            "Nested 'resources' section is no longer supported. "
            "Move resource keys to the top level of the config file."
        )
    if "tasks" in cfg and isinstance(cfg["tasks"], list):
        raise ValueError(
            "Inline 'tasks' in config is no longer supported. "
            "Pass a task file as the second argument."
        )
    if "tasks_file" in cfg:
        raise ValueError(
            "Config should only contain infrastructure. "
            "Pass the task file as the second argument."
        )
    if "work_base" in cfg:
        raise ValueError(
            "Config should only contain infrastructure. Use --work-base flag instead."
        )
    if "forward_common_files" in cfg:
        raise ValueError(
            "Config should only contain infrastructure. "
            "Use --common-files flag instead."
        )


def _load_task_dicts(tasks_path: str | Path) -> tuple[list[dict], str, list[str]]:
    """Load task definitions from a YAML file.

    Returns (task_dicts, work_base, forward_common_files).
    work_base defaults to "." and forward_common_files defaults to [].
    """
    tasks_path = Path(tasks_path)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    with open(tasks_path) as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict):
        if "command" in data:
            return [data], ".", []
        if "tasks" in data:
            tasks = data["tasks"]
            if not isinstance(tasks, list):
                raise ValueError(f"'tasks' key must contain a list: {tasks_path}")
            work_base = data.get("work_base", ".")
            fwd = data.get("forward_common_files", [])
            return tasks, work_base, fwd
        raise ValueError(
            f"Task file dict must have 'command' or 'tasks' key: {tasks_path}"
        )

    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Task file list item {i} is not a dict: {tasks_path}")
        return data, ".", []

    raise ValueError(
        f"Task file must be a YAML dict (single task) or list (multiple tasks): "
        f"{tasks_path}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="minidispatch CLI")
    parser.add_argument("config", help="Bridge config YAML file")
    parser.add_argument("tasks", help="Task YAML file")
    parser.add_argument(
        "--work-base",
        default=None,
        help="Experiment workspace name (default: '.')",
    )
    parser.add_argument(
        "--common-files",
        nargs="*",
        default=None,
        help="Shared input files to forward",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=None,
        help="Polling interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Max retries for failed jobs (default: 3)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean remote directory after completion",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _validate_config(cfg)

    # Build backend from flat config (picks batch_type, context_type, roots, profile)
    backend = Backend.from_dict(cfg)

    # Build resources from flat config
    resources = Resources(
        number_node=cfg.get("number_node", 1),
        cpu_per_node=cfg.get("cpu_per_node", 1),
        gpu_per_node=cfg.get("gpu_per_node", 0),
        queue_name=cfg.get("queue_name", ""),
        para_job=cfg.get("para_job", 0),
        module_purge=cfg.get("module_purge", False),
        module_list=cfg.get("module_list", []),
        module_unload_list=cfg.get("module_unload_list", []),
        source_list=cfg.get("source_list", []),
        prepend_script=cfg.get("prepend_script", []),
        append_script=cfg.get("append_script", []),
        envs=cfg.get("envs", {}),
        custom_flags=cfg.get("custom_flags", []),
        if_cuda_multi_devices=cfg.get("if_cuda_multi_devices", False),
        kwargs=cfg.get("kwargs", {}),
    )

    # Load tasks from separate file
    task_dicts, file_work_base, file_common_files = _load_task_dicts(args.tasks)

    # CLI flags override task file values
    work_base = args.work_base if args.work_base is not None else file_work_base
    common_files = (
        args.common_files if args.common_files is not None else file_common_files
    )

    tasks = [
        Task(
            command=t["command"],
            task_work_path=t["task_work_path"],
            forward_files=t.get("forward_files", []),
            backward_files=t.get("backward_files", []),
            outlog=t.get("outlog", "log"),
            errlog=t.get("errlog", "err"),
        )
        for t in task_dicts
    ]

    submission = Submission(
        work_base=work_base,
        backend=backend,
        resources=resources,
        task_list=tasks,
        forward_common_files=common_files,
    )

    # Resolve run options: CLI flag > config value > hardcoded default
    check_interval = args.check_interval if args.check_interval is not None else cfg.get("check_interval", 10)
    max_retries = args.max_retries if args.max_retries is not None else cfg.get("max_retries", 3)
    clean = False if args.no_clean else cfg.get("clean", True)

    submission.run(
        check_interval=check_interval,
        max_retries=max_retries,
        clean=clean,
    )


if __name__ == "__main__":
    main()
