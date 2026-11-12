"""
mobility.py  (Gauss-Markov mobility model)
==========================================
Generates each UE's movement using the GAUSS-MARKOV mobility model
(Liang & Haas, 1999) — a standard, widely-cited model. Its memory parameter
alpha in [0,1] controls how smooth/correlated the motion is:
  alpha -> 1 : nearly straight, highly correlated (smooth)
  alpha -> 0 : memoryless random walk

Speed and direction each follow:
  x_n = alpha*x_{n-1} + (1-alpha)*x_mean + sqrt(1-alpha^2)*noise

A `scenario` (from scenarios.py) sets the mean speed(s) and whether the mean
speed varies over time (non-stationary).
"""

import numpy as np

from config.config import CFG

STATIONARY, PEDESTRIAN, VEHICULAR = 0, 1, 2


def generate_ue_trajectory(rng, scenario) -> tuple:
    """Generate one UE's path under a scenario spec.

    Returns positions (num_steps,2), regimes (num_steps,), and the blockage
    multiplier to apply in the channel for this scenario.
    """
    m = CFG.mobility
    num_steps = int(m.sim_duration_s / m.time_step_s)
    area = CFG.network.area_size_m
    alpha = m.gauss_markov_alpha

    positions = np.zeros((num_steps, 2), dtype=np.float64)
    regimes = np.zeros(num_steps, dtype=np.int64)

    speeds = scenario.regime_speeds_ms
    mean_speed = float(rng.choice(speeds))
    speed = mean_speed
    heading = rng.uniform(0, 2 * np.pi)
    mean_heading = heading
    pos = rng.uniform(0, area, size=2)

    for t in range(num_steps):
        # occasionally change the target mean speed (regime switch -> non-stationary)
        if scenario.regime_switch_prob > 0 and rng.random() < scenario.regime_switch_prob:
            mean_speed = float(rng.choice(speeds))
        # non-stationary scenario: the mean speed also drifts smoothly over time
        if scenario.time_varying_speed:
            mean_speed = 5.0 + 15.0 * (0.5 + 0.5 * np.sin(np.pi * t * m.time_step_s / 120.0))

        # Gauss-Markov updates for speed and heading
        speed = (alpha * speed + (1 - alpha) * mean_speed
                 + np.sqrt(1 - alpha ** 2) * rng.normal(0, 1.0))
        speed = max(0.0, speed)
        heading = (alpha * heading + (1 - alpha) * mean_heading
                   + np.sqrt(1 - alpha ** 2) * rng.normal(0, 0.3))

        pos = pos + speed * m.time_step_s * np.array([np.cos(heading), np.sin(heading)])
        for d in range(2):   # reflect at the area edges
            if pos[d] < 0:
                pos[d] = -pos[d]; heading = np.pi - heading; mean_heading = heading
            elif pos[d] > area:
                pos[d] = 2 * area - pos[d]; heading = np.pi - heading; mean_heading = heading

        positions[t] = pos
        regimes[t] = (STATIONARY if speed < 0.5 else
                      PEDESTRIAN if speed < 5.0 else VEHICULAR)

    return positions, regimes, scenario.blockage_mult
