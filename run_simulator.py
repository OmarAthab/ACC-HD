"""
run_simulator.py  —  RUN THIS FIRST
===================================
Builds the whole 5G-Advanced handover dataset and saves it under results/dataset/.
You will see progress lines while it runs (percentage + estimated time left).

How to run in PyCharm:
  right-click this file -> Run 'run_simulator'
or from the Terminal tab:
  python run_simulator.py

Nothing here needs the internet or a GPU. Batch 1 uses only numpy/pandas/matplotlib.
"""

import sys
from pathlib import Path

# Make sure the project root is importable when run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CFG
from common.seeding import set_all_seeds
from simulator.dataset import build_all


def main() -> None:
    print("=" * 60)
    print(" UA-PHM  |  building scenario datasets (source + shifted)")
    print("=" * 60)
    set_all_seeds(CFG.seed)          # reproducible run
    build_all()
    print("\nAll done. Organized datasets are in:", CFG.data_dir)
    print("Next: run experiments/run_all.py")


if __name__ == "__main__":
    main()
