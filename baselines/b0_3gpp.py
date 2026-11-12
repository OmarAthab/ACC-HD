"""
b0_3gpp.py  —  Baseline B0: conventional 3GPP A3 / Conditional Handover
=======================================================================
The non-ML reference every handover paper includes. It does NOT learn: it simply
predicts that the future serving cell is the currently strongest cell (the cell
with the highest RSRP at the most recent measurement step), which is what an
event-triggered A3/CHO scheme effectively does.

Our feature vector per step is [RSRP(all cells), SINR(all cells)] (normalised).
So the first `num_cells` values of the LAST step are the (normalised) RSRPs, and
the strongest cell = their argmax.
"""

import numpy as np


def predict(X, num_cells):
    """Return predicted target-cell probabilities (one-hot at the strongest cell).

    X: (N, window_len, 2*num_cells). Returns (N, num_cells) 'probabilities'
    (a hard 1.0 at the predicted cell) so it can go through the same metrics.
    """
    X = np.asarray(X)
    last_step = X[:, -1, :num_cells]              # normalised RSRP of each cell
    pred = last_step.argmax(axis=1)               # strongest cell now
    probs = np.zeros((len(X), num_cells), dtype=float)
    probs[np.arange(len(X)), pred] = 1.0
    return probs
