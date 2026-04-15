from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


def make_run_dir(output_root: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_file(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
