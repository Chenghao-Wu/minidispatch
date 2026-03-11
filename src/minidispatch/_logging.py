from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("minidispatch")
log.propagate = False
log.setLevel(logging.INFO)

# File handler — cwd first, fallback to home
cwd_logfile = os.path.join(os.getcwd(), "minidispatch.log")
_fh = logging.FileHandler(cwd_logfile, delay=True)
try:
    log.addHandler(_fh)
    log.info(f"Log file: {cwd_logfile}")
except PermissionError:
    log.removeHandler(_fh)
    home_logfile = os.path.join(os.path.expanduser("~"), "minidispatch.log")
    _fh = logging.FileHandler(home_logfile, delay=True)
    log.addHandler(_fh)
    log.info(f"Log file: {home_logfile}")

_formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")
_fh.setFormatter(_formatter)

# Console handler — stdout (tqdm uses stderr)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_formatter)
log.addHandler(_sh)
