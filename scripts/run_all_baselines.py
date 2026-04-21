from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def run(script_name: str) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    run("prepare_datasets.py")
    run("train_regression_baseline.py")
    run("train_size_classification_baseline.py")
    run("train_sex_classification_baseline.py")


if __name__ == "__main__":
    main()
