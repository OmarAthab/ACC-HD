"""
scenarios.py
============
Isolated mobility/channel scenarios for the distribution-shift protocol.

Design (genuine out-of-source shift): the SOURCE scenario is a narrow, benign
urban environment (pedestrian and slow-vehicular speeds, nominal blockage). Its
UEs are split by UE into train/val/calib/test. The shifted scenarios are
generated with statistics NOT present in the source -- higher speed, heavier
blockage, and non-stationary drift -- so they are genuinely out-of-source
distribution shifts (not marginals of a mixed source). Each shifted scenario is
used entirely as a held-out test set; no UE is shared across scenarios or splits.

Roles:
  urban            -> SOURCE (train / val / calibration / in-distribution test);
                      slow, low-switch, nominal blockage.
  vehicular_fast   -> SHIFT: higher speed unseen in the source.
  high_blockage    -> SHIFT: much heavier blockage than the source.
  non_stationary   -> SHIFT: time-varying speed spanning beyond the source + blockage.
"""

from dataclasses import dataclass


@dataclass
class ScenarioSpec:
    name: str
    regime_speeds_ms: list          # candidate mean speeds (m/s) the UE can take
    regime_switch_prob: float       # chance per step of switching regime
    time_varying_speed: bool = False  # non-stationary: mean speed drifts over time
    blockage_mult: float = 1.0      # multiplier on the channel blockage probability


SCENARIOS = {
    # SOURCE distribution: narrow, benign urban (pedestrian + slow vehicular)
    "urban": ScenarioSpec("urban", regime_speeds_ms=[1.4, 8.0],
                          regime_switch_prob=0.003, blockage_mult=1.0),

    # shifted test distributions (statistics UNSEEN in the source)
    "vehicular_fast": ScenarioSpec("vehicular_fast", regime_speeds_ms=[20.0],
                                   regime_switch_prob=0.0, blockage_mult=1.0),
    "high_blockage": ScenarioSpec("high_blockage", regime_speeds_ms=[1.4, 8.0],
                                  regime_switch_prob=0.003, blockage_mult=3.0),
    "non_stationary": ScenarioSpec("non_stationary", regime_speeds_ms=[1.4, 8.0, 20.0],
                                   regime_switch_prob=0.006,
                                   time_varying_speed=True, blockage_mult=2.0),
}

SOURCE_SCENARIO = "urban"
# ordered easy -> hard for the drifting-stream experiment
SHIFT_SCENARIOS = ["vehicular_fast", "high_blockage", "non_stationary"]
