# minidispatch

minidispatch bridges where you **prepare** work (local machine) and where it **runs** (compute node — same machine or remote HPC cluster).

```
Local Machine                          Compute Node
(prepare work)                         (run jobs)
─────────────                          ────────────
local_root/                            remote_root/
  task_000/                              task_000/
    train.py        ── upload ──►          train.py
    data.csv                               data.csv
                                           model.pt  (generated)
    model.pt        ◄── download ──        log
    log
```

The flow:

1. **Generate jobs** — group tasks and allocate GPUs
2. **Upload** forward files from `local_root` to `remote_root`
3. **Submit & poll** — launch batch scripts and wait for completion
4. **Download** backward files from `remote_root` back to `local_root`
5. **Recovery** — if interrupted, resume from saved state on next run

## Install

```sh
pip install .

# With SSH support
pip install ".[ssh]"
```

## Quick Start

Define your tasks once, then choose where they run:

```python
from minidispatch import Backend, Resources, Submission, Task

tasks = [
    Task(
        command="python train.py",
        task_work_path=f"task_{i:03d}",
        forward_files=["train.py", "data.csv"],
        backward_files=["model.pt", "log"],
    )
    for i in range(4)
]

resources = Resources(
    gpu_per_node=2,
    if_cuda_multi_devices=True,
    module_list=["cuda/11.0"],
)
```

### Local bridge — Shell + LocalContext

Both directories on the same machine. Good for workstations with GPUs.

```python
from minidispatch.contexts import LocalContext

context = LocalContext(local_root="./project", remote_root="/tmp/dispatch")
backend = Backend.create("shell", context)

submission = Submission(
    work_base="calculations/",
    backend=backend,
    resources=resources,
    task_list=tasks,
)
submission.run(check_interval=5)
```

### Remote bridge — Slurm + SSHContext

Local laptop prepares work, HPC cluster runs it.

```python
from minidispatch.contexts import SSHContext

context = SSHContext(
    local_root="./project",
    remote_root="/scratch/user/dispatch",
    remote_profile={"hostname": "hpc.example.com", "username": "user"},
)
backend = Backend.create("slurm", context)

submission = Submission(
    work_base="calculations/",
    backend=backend,
    resources=resources,
    task_list=tasks,
)
submission.run(check_interval=30)
```

## CLI

```sh
minidispatch config.yaml tasks.yaml
minidispatch config.yaml tasks.yaml --work-base my_experiment
minidispatch config.yaml tasks.yaml --common-files input.dat
minidispatch config.yaml tasks.yaml --check-interval 10 --max-retries 3
```

See `examples/` for config and task file formats.

## Downloading Task Outputs

The `backward_files` parameter on each `Task` controls which files are downloaded from the remote working directory after the task completes. Only files matching the listed patterns are copied back.

```python
# Download specific files
Task(command="python train.py", backward_files=["model.pt", "log"])

# Download all top-level files and directories (directories copied recursively)
Task(command="python train.py", backward_files=["*"])
```

**Notes:**
- `["*"]` matches all top-level files and directories; directories are copied recursively.
- Hidden files (names starting with `.`) are not matched by `*`. Add `".*"` to include them.

## Features

- **Backends**: Shell (local), Slurm
- **Contexts**: Local filesystem, SSH (paramiko)
- **Recovery**: JSON-based crash recovery with corruption resilience
- **CUDA**: Per-job GPU assignment via `CUDA_VISIBLE_DEVICES`
- **Job throttling**: Limit concurrent submitted jobs via `para_job`
- **Modules**: `module purge/load/unload` and `source` support
- **Progress**: tqdm bars for upload, download, and job polling
