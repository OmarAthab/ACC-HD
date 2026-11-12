"""
events.py
=========
From the signal strengths, decide:
  - the SERVING cell at each step (the strongest cell, with a little hysteresis
    so the UE does not flip back and forth on tiny fluctuations), and
  - the PREDICTION TARGET: the serving cell H steps into the future (this is the
    label the predictor tries to forecast).

It also marks the true handover instants (when the serving cell changes).
"""

import numpy as np

from config.config import CFG

# Real handover uses BOTH a hysteresis margin (a challenger cell must be
# clearly better) AND a time-to-trigger (it must stay better for several steps).
# Together these make the serving cell "sticky", exactly as in 3GPP networks,
# which is what makes the future serving cell predictable.
HYSTERESIS_DB = 4.0     # a challenger must beat the serving cell by this many dB
TIME_TO_TRIGGER = 5     # ...for this many consecutive steps before we switch


def compute_serving_cells(rsrp_dbm: np.ndarray) -> np.ndarray:
    """Return the serving-cell index at each step, shape (num_steps,).

    Uses hysteresis + time-to-trigger so the serving cell does not flip on brief
    signal fluctuations (avoids ping-pong; mirrors real A3/CHO behaviour).
    """
    num_steps, num_cells = rsrp_dbm.shape
    serving = np.zeros(num_steps, dtype=np.int64)

    current = int(np.argmax(rsrp_dbm[0]))
    serving[0] = current
    challenger = -1
    counter = 0
    for t in range(1, num_steps):
        best = int(np.argmax(rsrp_dbm[t]))
        beats = best != current and \
            (rsrp_dbm[t, best] - rsrp_dbm[t, current]) > HYSTERESIS_DB
        if beats:
            if best == challenger:
                counter += 1
            else:
                challenger, counter = best, 1
            if counter >= TIME_TO_TRIGGER:      # sustained -> execute handover
                current, challenger, counter = best, -1, 0
        else:
            challenger, counter = -1, 0          # challenger no longer strong
        serving[t] = current
    return serving


def compute_prediction_targets(serving: np.ndarray, horizon: int) -> np.ndarray:
    """Label at step t = serving cell at step (t + horizon).

    The last `horizon` steps have no future label, so they are marked -1 and
    dropped later when building samples.
    """
    num_steps = serving.shape[0]
    targets = np.full(num_steps, -1, dtype=np.int64)
    if num_steps > horizon:
        targets[:num_steps - horizon] = serving[horizon:]
    return targets


def handover_mask(serving: np.ndarray) -> np.ndarray:
    """Boolean array: True at steps where the serving cell changed (a handover)."""
    changed = np.zeros_like(serving, dtype=bool)
    changed[1:] = serving[1:] != serving[:-1]
    return changed
