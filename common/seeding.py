"""
seeding.py
==========
One function that fixes every random-number generator we use, so that each run
produces exactly the same results (reproducibility requirement).

Call `set_all_seeds(SEED)` once at the very start of any script.
"""

import os
import random
import numpy as np


def set_all_seeds(seed: int) -> None:
    """Fix Python, NumPy (and PyTorch if installed) random seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch is only present from Batch 2 onward; import it lazily so Batch 1
    # runs with numpy alone.
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make GPU math deterministic too (slightly slower, fully repeatable).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch not installed (or not fully installed) — fine for numpy-only batches.
        pass
