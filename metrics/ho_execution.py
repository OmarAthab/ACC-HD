"""
ho_execution.py  -  realistic temporal handover-execution KPIs (3GPP-faithful)
==============================================================================
Turns a method's per-decision score rows into an EXECUTED handover sequence over
time (per UE) under a REALISTIC handover-execution model, and reports the
standard 5G handover KPIs -- not prediction-referenced proxies.

Every method (proposed and baselines alike) is driven through the SAME execution
model so the comparison is apples-to-apples and faithful to how a handover is
actually triggered and completed in a 3GPP network:

  * TRIGGER (uniform, method-agnostic).  A handover toward the method's top
    candidate cell is committed only when that candidate (i) differs from the
    serving cell, (ii) is preferred by the method over the serving cell by a
    hysteresis margin, and (iii) stays the top candidate for TTT (time-to-trigger)
    consecutive decision steps.  This is the standard A3 stabilisation
    (hysteresis + TTT) that stops a controller ping-ponging on flickering
    decisions -- applied identically to all methods.  It gates on each method's
    OWN output stream, so a predictive method may still act proactively (it is
    NOT forced to wait for a measured A3 event); it is only prevented from
    thrashing.  The PROPOSED method additionally gates on its calibrated
    confidence (HOLD -> defer), and, when it commits, prepares its CONFORMAL
    PREDICTION SET as the conditional-handover (CHO) candidate list instead of a
    single target -- see below.  Those two things are its contribution and the
    only difference from acting on the raw predictions.

  * CONDITIONAL HANDOVER WITH THE CONFORMAL SET (proposed only).  3GPP CHO
    prepares SEVERAL candidate cells and executes to whichever satisfies the
    execution condition.  When `cand_sets` is given, a committed handover prepares
    every cell in that step's conformal set (which, by construction, contains the
    true serving cell with probability >= 1-alpha) and completes to the realised
    serving cell if it was prepared, else to the best prepared cell that becomes
    servable.  It thus fails only if NO prepared cell becomes servable in the
    window.  Baselines have no calibrated set, so they prepare a single target.
    The mean number of prepared cells (`mean_prepared`) is the CHO signalling
    overhead and is reported alongside, so the reliability gain is shown with its
    cost.

  * SUCCESS / HANDOVER FAILURE (HOF) is PHYSICAL, over the arrival window.  A
    committed handover succeeds when the target cell becomes servable
    (measured SINR >= Q_out) within the ARRIVAL WINDOW -- from the decision
    instant up to the prepared handover's execution a few steps later
    (`arrival_window_steps`, sized to the prediction horizon).  Otherwise it is a
    Handover Failure: the prepared target never becomes usable, so the handover
    does not complete and the UE must re-handover.  Checking a WINDOW (not the
    instantaneous SINR) is what makes this fair to BOTH families: a PREDICTIVE
    handover to a cell that is weak now but strong H steps later still succeeds
    (its target becomes servable inside the window), while a REACTIVE handover to
    the currently strongest cell also succeeds (it is servable immediately).  It
    is measured identically for every method and is the true 3GPP HOF semantics
    (target cannot sustain the link at execution).  The RLF path is likewise
    fully physical.

  * HANDOVER INTERRUPTION TIME (HIT) reflects timing.  A successful handover
    performed while the SOURCE link is still healthy (source SINR >= Q_out) is
    proactive -> CHO/LTM band (~20-30 ms).  One performed after the source has
    already degraded is late/reactive -> legacy band (~50-90 ms).  A failed
    handover pays a penalty.  Predictive, early handovers therefore earn the low
    interruption band; late reactive ones do not.

  * RADIO LINK FAILURE (RLF) + RRC RE-ESTABLISHMENT.  If the serving-cell SINR
    stays below Q_out for T310 consecutive steps, an RLF is counted AND the UE
    performs an RRC re-establishment to the strongest measurable cell (it
    recovers -- it does not stay stuck), paying a re-establishment interruption.
    This bounds RLF realistically and stops a deferred/failed UE from accruing an
    unbounded number of failures on one dead cell.

  * PING-PONG: a successful handover back to the immediately previous serving
    cell within T_crit steps.  TIME-OF-STAY (ToS): mean time between handovers.

Inputs are the per-sample arrays saved by dataset.py (raw SINR in dB, current
serving cell, UE id -- all in temporal order within a UE) plus the method's
per-cell score row.  Pure NumPy; identical for every method.
"""

import numpy as np

from config.config import CFG


def compute_ho_kpis(sinr_raw, serving_now, ue, scores, confident=None,
                    cand_sets=None, y_true=None, step_seconds=None):
    """Replay each UE in time order under the realistic execution model.

    Parameters
    ----------
    sinr_raw : (N, K) float   raw measured SINR (dB) of every cell at each step.
    serving_now : (N,) int    the UE's serving cell at each step.
    ue : (N,) int             UE id (samples are time-ordered within a UE).
    scores : (N, K) float     the method's per-cell score/probability row.
    cand_sets : (N, K) bool or None
        The conformal prediction set per step (proposed method): a committed
        handover prepares all set members (CHO). None -> single-target handover.
    y_true : (N,) int or None
        The realised serving cell H steps ahead; used only to land a CHO on the
        cell the UE actually moves into when that cell was prepared and servable.
    confident : (N,) bool or None
        Per-step commit permission (the proposed method's confidence gate: True
        where the policy allows a handover, False where it defers/HOLDs).
        None -> every step is commit-eligible (baselines have no confidence gate).
    step_seconds : float or None   seconds represented by one decision step.
    """
    hx = CFG.hexec
    ttt = int(getattr(hx, "ttt_steps", 3))
    hyst = float(getattr(hx, "a3_hysteresis", 0.05))
    w_arr = int(getattr(hx, "arrival_window_steps", 5))
    if step_seconds is None:
        step_seconds = CFG.task.window_stride * CFG.mobility.time_step_s

    sinr_raw = np.asarray(sinr_raw, dtype=float)
    serving_now = np.asarray(serving_now).astype(int)
    ue = np.asarray(ue).astype(int)
    scores = np.asarray(scores, dtype=float)
    K = scores.shape[1]
    if confident is None:
        confident = np.ones(len(ue), dtype=bool)
    else:
        confident = np.asarray(confident).astype(bool)
    if cand_sets is not None:
        cand_sets = np.asarray(cand_sets).astype(bool)
    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)

    cand_all = scores.argmax(axis=1)

    total_ho = 0        # committed (voluntary) handovers
    total_hof = 0       # of those, ones whose target could not hold the link
    total_pp = 0        # of those, ones returning to the previous cell within T_crit
    total_rlf = 0       # radio-link failures
    total_reest = 0     # forced RRC re-establishments (recoveries)
    total_prep = 0      # summed prepared candidates over committed handovers (CHO cost)
    hit_sum = 0.0       # summed interruption over HANDOVER events only (LTM/legacy/failed)
    hit_events = 0
    recovery_sum = 0.0  # summed RLF re-establishment time (kept SEPARATE from HIT)
    tos_list = []
    n_ue = 0

    for u in np.unique(ue):
        idx = np.where(ue == u)[0]           # already time-ordered within a UE
        if len(idx) == 0:
            continue
        n_ue += 1
        cur = int(serving_now[idx[0]])
        prev = cur
        pending_tgt = -1
        pending_cnt = 0
        steps_since_ho = 0
        below = 0

        for p, i in enumerate(idx):
            s_src = float(sinr_raw[i, cur]) if cur < K else -100.0

            # ---- radio-link-failure tracking + RRC re-establishment ----
            if s_src < hx.q_out_sinr_db:
                below += 1
                if below >= hx.t310_steps:
                    total_rlf += 1
                    best = int(np.argmax(sinr_raw[i]))     # recover to strongest cell
                    if best != cur:
                        prev, cur = cur, best
                        total_reest += 1
                    recovery_sum += hx.rlf_recovery_ms     # RRC re-establishment (separate from HIT)
                    below = 0
                    steps_since_ho = 0
                    pending_tgt = -1
                    pending_cnt = 0
                    continue
            else:
                below = 0

            # ---- uniform A3-style trigger: hysteresis margin + TTT ----
            cand = int(cand_all[i])
            margin = float(scores[i, cand] - (scores[i, cur] if cur < K else -1e9))
            want = (cand != cur) and (margin >= hyst)
            if want and cand == pending_tgt:
                pending_cnt += 1
            elif want:
                pending_tgt = cand
                pending_cnt = 1
            else:
                pending_tgt = -1
                pending_cnt = 0

            # minimum time-of-stay (ping-pong avoidance): do not hand back to the
            # just-left cell within T_crit steps -- applied uniformly to all methods.
            trigger = (want and (pending_cnt >= ttt) and bool(confident[i])
                       and not (cand == prev and steps_since_ho <= hx.tcrit_steps))

            if trigger:
                tgt = cand
                total_ho += 1
                fut = idx[p:min(len(idx), p + 1 + w_arr)]
                # prepared candidate list: the conformal set (CHO) or a single target
                if cand_sets is not None and cand_sets[i].any():
                    prepared = list(np.where(cand_sets[i])[0])
                    if tgt not in prepared:
                        prepared.append(tgt)
                else:
                    prepared = [tgt]
                total_prep += len(prepared)
                # physical execution: succeeds if a prepared cell becomes servable
                # within the arrival window (decision .. prepared execution).
                servable = [c for c in prepared
                            if c < K and bool(np.any(sinr_raw[fut, c] >= hx.q_out_sinr_db))]
                if servable:                                # handover completes -> success
                    # min time-of-stay: avoid landing on the just-left cell within T_crit
                    elig = [c for c in servable
                            if not (c == prev and steps_since_ho <= hx.tcrit_steps)] or servable
                    if y_true is not None and int(y_true[i]) in elig:
                        land = int(y_true[i])               # UE moves into its realised cell
                    else:
                        pool = [c for c in elig if c != prev] or elig
                        land = max(pool, key=lambda c: float(np.max(sinr_raw[fut, c])))
                    # make-before-break (LTM) if the source is still comfortable at
                    # execution (proactive), else a late/reactive L3-style handover.
                    hit_sum += (hx.hit_ltm_ms if s_src >= hx.proactive_margin_db
                                else hx.hit_legacy_ms)
                    hit_events += 1
                    if land == prev and steps_since_ho <= hx.tcrit_steps:
                        total_pp += 1
                    tos_list.append(steps_since_ho)
                    prev, cur = cur, land
                    steps_since_ho = 0
                    below = 0
                else:                                       # no prepared cell servable -> HOF
                    total_hof += 1
                    hit_sum += hx.hit_failed_ms
                    hit_events += 1
                    steps_since_ho += 1
                pending_tgt = -1
                pending_cnt = 0
            else:
                steps_since_ho += 1

    tho = max(1, total_ho)
    return {
        "hof_rate": total_hof / tho,                        # fraction of committed HOs that fail
        "ping_pong_rate": total_pp / tho,                   # fraction returning within T_crit
        "rlf_per_ue": total_rlf / max(1, n_ue),             # radio-link failures per UE
        "reestab_per_ue": total_reest / max(1, n_ue),       # forced RRC re-establishments per UE
        "mean_hit_ms": hit_sum / max(1, hit_events),        # mean HANDOVER interruption time (ms)
        "mean_recovery_ms": (recovery_sum / total_rlf) if total_rlf else 0.0,  # mean RLF recovery (ms), separate
        "mean_tos_s": (float(np.mean(tos_list)) * step_seconds) if tos_list else float("nan"),
        "ho_per_ue": total_ho / max(1, n_ue),               # committed handovers per UE
        "mean_prepared": total_prep / tho,                  # avg prepared candidates (CHO overhead)
    }
