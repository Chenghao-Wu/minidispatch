# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

minidispatch is a simplified job dispatch library with crash recovery, CUDA multi-device support, and tqdm progress bars. Inspired by dpdispatcher. It dispatches shell commands as batch jobs to local or remote machines, handles file transfer, and polls for completion.

## Commands

```sh
# Install
pip install -e ".[dev]"

# Run tests
pytest
pytest tests/test_models.py              # single file
pytest tests/test_shell.py::TestShellIntegration::test_single_task  # single test

# Lint and type check
ruff check src/ tests/
ruff format --check src/ tests/
pyright
```

## Architecture

The library has a two-axis plugin design: **Backends** (how jobs run) x **Contexts** (where files live and commands execute).

### Plugin Registry Pattern

Both `Backend` and `BaseContext` use `__init_subclass__` auto-registration. Subclasses register themselves by class name (lowercased) when imported. The `__init__.py` files in `backends/` and `contexts/` trigger these imports. Factory methods: `Backend.create("shell", ctx)` and `BaseContext.create("local", ...)`.

### Core Flow (Submission.run)

1. `generate_jobs()` — shuffles tasks (seed=42) and groups them by `Resources.group_size`
2. `try_recover()` — loads state from `{submission_hash}.json` if it exists (atomic write via tmp+rename)
3. Upload forward files via context
4. Submit jobs via backend (`do_submit` generates bash scripts and launches them)
5. Poll loop — `check_status` + `_handle_job_states` (resubmits terminated jobs up to `max_retries`)
6. Download backward files via context

### Key Classes

- **`Submission`** (`submission.py`) — orchestrator: groups tasks into jobs, manages the run loop, handles recovery
- **`Job`** (`job.py`) — a group of `Task`s submitted as one batch script. Keyed by `job_hash` (SHA1 of serialized tasks+resources)
- **`Task` / `Resources`** (`models.py`) — data classes. Task = single command + file lists. Resources = compute requirements + environment setup
- **`Backend`** (`backend.py`) — ABC for batch systems. Generates bash scripts from templates, handles submit/status/kill. Implementations: `Shell` (local nohup), `Slurm` (sbatch)
- **`BaseContext`** (`context.py`) — ABC for file I/O and command execution. Implementations: `LocalContext` (filesystem copy + subprocess), `SSHContext` (paramiko SFTP + exec_command)

### Script Generation

`Backend.gen_script()` assembles bash scripts from templates in `backend.py`. The script structure: header -> custom flags -> env setup (modules, exports) -> per-task commands (with CUDA device assignment) -> wait+finish tag. Shell and Slurm both override `gen_script` to source a separate `.run` file for commands.

### CUDA Multi-Device

When `Resources.if_cuda_multi_devices=True`, tasks get round-robin `CUDA_VISIBLE_DEVICES` assignment. The mutable counters `gpu_in_use` and `task_in_para` on Resources track assignment during script generation.

## Conventions

- Python >=3.9, uses `from __future__ import annotations`
- Ruff for linting (rules: E, F, I, W) with line-length 88
- Pyright for type checking
- SSH support is optional (`paramiko`); SSHContext import is guarded with try/except
- Tests use `pytest` with class-based grouping; SSH tests are skipped without paramiko
- All serialization uses plain dicts with `serialize()`/`deserialize()` class methods
