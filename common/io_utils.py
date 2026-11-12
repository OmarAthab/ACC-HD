"""
io_utils.py
===========
Small helpers to save and load data and results, so every module uses the same
file formats and folders. Keeping this in one place means a beginner can find
"where do files get written?" in a single spot.

- Arrays (the dataset) are saved as compressed .npz files.
- Tables of results are saved as .csv (open in Excel) and .json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create a folder (and parents) if it does not exist yet."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_arrays(path: Path, **arrays) -> None:
    """Save named numpy arrays into one compressed .npz file.

    Example: save_arrays(dir/'data.npz', features=X, labels=y)
    """
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)
    print(f"  saved arrays -> {path}", flush=True)


def load_arrays(path: Path) -> dict:
    """Load a .npz file back into a dict of arrays."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def save_table(path_no_ext: Path, rows: list) -> None:
    """Save a list of dict rows as BOTH .csv and .json (same base name)."""
    ensure_dir(path_no_ext.parent)
    df = pd.DataFrame(rows)
    df.to_csv(path_no_ext.with_suffix(".csv"), index=False)
    with open(path_no_ext.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  saved table -> {path_no_ext}.csv / .json", flush=True)
