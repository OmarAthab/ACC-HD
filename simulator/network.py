"""
network.py
==========
Builds the fixed network layout: the (x, y) positions of the base stations (cells).

Layout: the standard 3GPP hexagonal cell cluster of 19 sites (a centre cell plus
two surrounding rings of 6 and 12), the layout used in 3GPP TR 38.901 for Urban
Micro/Macro evaluations. With the Urban Micro inter-site distance (ISD = 200 m,
TR 38.901) the cluster spans roughly 1.2 km x 1.2 km. Each of the 19 sites is
one candidate handover cell; channel.py applies the 3GPP three-sector antenna
pattern (best-oriented sector) only to compute a realistic per-site SINR.
"""

import numpy as np

from config.config import CFG


def build_cell_positions() -> np.ndarray:
    """Return (num_cells, 2) metres for a symmetric hexagonal 19-cell cluster.

    Sites are placed on a triangular (hex) lattice at all axial coordinates within
    hex-distance 2 of the centre (1 + 6 + 12 = 19 sites), spaced by the inter-site
    distance, then centred in the coverage area.
    """
    n = CFG.network.num_cells
    isd = CFG.network.inter_site_distance_m
    area = CFG.network.area_size_m

    # axial hex coordinates (q, r) with cube distance <= 2 from the centre
    coords = []
    ring = 2
    for q in range(-ring, ring + 1):
        for r in range(-ring, ring + 1):
            if (abs(q) + abs(r) + abs(q + r)) / 2 <= ring:
                coords.append((q, r))
    coords.sort(key=lambda c: (abs(c[0]) + abs(c[1]) + abs(c[0] + c[1])) / 2)  # centre-out
    coords = coords[:n]

    pos = np.array([[isd * (q + r / 2.0), isd * (np.sqrt(3) / 2.0 * r)]
                    for (q, r) in coords], dtype=np.float64)

    # centre the cluster inside the coverage area
    pos[:, 0] += area / 2.0 - pos[:, 0].mean()
    pos[:, 1] += area / 2.0 - pos[:, 1].mean()
    return pos
