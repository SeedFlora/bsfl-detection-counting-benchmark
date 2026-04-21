from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from larvae_cv.model_benchmarks import run_regression_and_size_benchmarks


def main() -> None:
    results = run_regression_and_size_benchmarks()
    print(results)


if __name__ == "__main__":
    main()
