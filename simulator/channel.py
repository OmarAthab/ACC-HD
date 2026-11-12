"""
channel.py  -  3GPP TR 38.901 Urban Micro (UMi) Street-Canyon channel
=====================================================================
Turns the geometry (UE vs. base-station positions) into per-cell RSRP and SINR
using the standard 3GPP TR 38.901 UMi model, rather than a generic log-distance
stand-in. Implemented (UMi Street Canyon, fc = 3.5 GHz):

  * 3D geometry: BS height 10 m, UE height 1.5 m -> 3D distance and elevation.
  * LOS probability (TR 38.901 Table 7.4.2-1) with distance-driven, temporally
    consistent LOS/NLOS state (no per-step flicker).
  * LOS and NLOS path loss (Table 7.4.1-1), with the breakpoint distance.
  * Shadow fading with scenario-correct std (LOS 4 dB, NLOS 7.82 dB), AR(1)
    time-correlated so the future serving cell stays predictable.
  * 3-sector sites with the 3GPP antenna element pattern (Table 7.3-1): 65 deg
    HPBW in azimuth and elevation, 8 dBi max gain, 30 dB front-to-back / side-lobe
    floor, mechanical downtilt. Each site serves a UE through its best-oriented
    sector; interference is the sum over the other sites' best-oriented sectors.
    The realistic SINR level thus comes from the geometry + antenna pattern, NOT
    from an ad-hoc interference scaling.

The 19 sites remain the handover cells (sector antennas are used only to compute
a realistic SINR). Pure NumPy; returns (RSRP_dBm, SINR_dB), both (steps, cells).
"""

import numpy as np

from config.config import CFG

_C = 3.0e8


def _los_probability(d2d):
    """TR 38.901 Table 7.4.2-1, UMi Street Canyon."""
    d2d = np.maximum(d2d, 1.0)
    p = 18.0 / d2d + np.exp(-d2d / 36.0) * (1.0 - 18.0 / d2d)
    return np.where(d2d <= 18.0, 1.0, p)


def _pathloss_los(d3d, d2d, fc_ghz, h_bs, h_ue, h_e):
    """UMi LOS path loss (dB), Table 7.4.1-1, with breakpoint distance."""
    fc_hz = fc_ghz * 1e9
    d_bp = 4.0 * (h_bs - h_e) * (h_ue - h_e) * fc_hz / _C          # breakpoint (m)
    pl1 = 32.4 + 21.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
    pl2 = (32.4 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
           - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ue) ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def _pathloss_nlos(d3d, pl_los, fc_ghz, h_ue):
    """UMi NLOS path loss (dB): max(LOS, NLOS'), Table 7.4.1-1."""
    pl_nlos = (35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc_ghz)
               - 0.3 * (h_ue - 1.5))
    return np.maximum(pl_los, pl_nlos)


def _antenna_gain_db(az_deg, el_deg):
    """3GPP element pattern (Table 7.3-1) for the BEST of `num_sectors` sectors.

    az_deg, el_deg: (T, K) azimuth/elevation from each site to the UE.
    Returns the best-sector gain (dBi), shape (T, K).
    """
    ch = CFG.channel
    hpbw = ch.ant_hpbw_deg
    a_m = ch.ant_front_back_db
    boresights = np.arange(ch.num_sectors) * (360.0 / ch.num_sectors)   # 0,120,240
    off = az_deg[..., None] - boresights                                # (T,K,S)
    off = (off + 180.0) % 360.0 - 180.0
    a_h = -np.minimum(12.0 * (off / hpbw) ** 2, a_m)                    # (T,K,S)
    a_v = -np.minimum(12.0 * ((el_deg + ch.ant_downtilt_deg) / hpbw) ** 2, a_m)  # (T,K)
    gain = ch.ant_gain_max_dbi + np.maximum(a_h + a_v[..., None], -a_m)  # (T,K,S)
    return gain.max(axis=-1)                                             # (T,K)


def compute_rsrp_sinr(ue_positions, cell_positions, rng, blockage_mult=1.0):
    """Return (RSRP_dBm, SINR_dB), each (num_steps, num_cells)."""
    ch = CFG.channel
    net = CFG.network
    h_bs, h_ue, h_e = ch.bs_height_m, ch.ue_height_m, ch.env_height_m
    fc = ch.carrier_freq_ghz

    ue = np.asarray(ue_positions, dtype=float)                 # (T,2)
    cells = np.asarray(cell_positions, dtype=float)            # (K,2)
    T, K = ue.shape[0], cells.shape[0]

    diff = ue[:, None, :] - cells[None, :, :]                  # (T,K,2)  (site -> UE)
    d2d = np.maximum(np.sqrt((diff ** 2).sum(axis=2)), 1.0)    # (T,K)
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ue) ** 2)               # (T,K)

    az = np.degrees(np.arctan2(diff[:, :, 1], diff[:, :, 0]))  # (T,K) azimuth site->UE
    el = np.degrees(np.arctan2(-(h_bs - h_ue), d2d))           # (T,K) elevation (downward)
    gain = _antenna_gain_db(az, el)                            # (T,K) best-sector gain

    # temporally/spatially consistent LOS state: one uniform field per (UE,cell),
    # thresholded by the distance-dependent LOS probability at each step.
    p_los = _los_probability(d2d)                              # (T,K)
    u_field = rng.random(K)[None, :]                           # (1,K) fixed for this UE
    is_los = u_field < p_los                                   # (T,K) bool

    pl_los = _pathloss_los(d3d, d2d, fc, h_bs, h_ue, h_e)
    pl = np.where(is_los, pl_los, _pathloss_nlos(d3d, pl_los, fc, h_ue))

    # shadow fading: AR(1) unit-variance process scaled by LOS/NLOS-dependent std
    rho = ch.shadowing_time_corr
    z = np.zeros((T, K))
    z[0] = rng.normal(0.0, 1.0, size=K)
    innov = np.sqrt(1.0 - rho ** 2)
    for t in range(1, T):
        z[t] = rho * z[t - 1] + rng.normal(0.0, innov, size=K)
    sf = z * np.where(is_los, ch.sf_los_db, ch.sf_nlos_db)

    fast = rng.normal(0.0, ch.fast_fading_std_db, size=(T, K))
    blocked = rng.random((T, K)) < (ch.blockage_prob * blockage_mult)
    blockage_loss = blocked * ch.blockage_extra_loss_db

    rsrp_dbm = net.cell_tx_power_dbm + gain - pl - sf - fast - blockage_loss

    # SINR: signal = each site's power; interference = load x sum of the OTHER
    # sites' (best-sector) power; noise = thermal floor.
    rsrp_lin = 10.0 ** (rsrp_dbm / 10.0)
    noise_lin = 10.0 ** (ch.noise_floor_dbm / 10.0)
    total_lin = rsrp_lin.sum(axis=1, keepdims=True)
    interference_lin = ch.load_factor * (total_lin - rsrp_lin)
    sinr_db = 10.0 * np.log10(np.maximum(rsrp_lin / (interference_lin + noise_lin), 1e-12))

    return rsrp_dbm, sinr_db
