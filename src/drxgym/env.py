"""Gymnasium environment: a gNB-side controller picks the UE's DRX config each window.

Each step is one decision window (default 1 s). The agent observes traffic and
QoS statistics from the *previous* window (what a base station could measure)
and chooses a (inactivity timer, cycle length) pair for the next window.

Reward = -(normalised energy) - lambda_latency * (fraction of packets over budget)

Energy is normalised by the energy of staying awake for the whole window, so
"DRX disabled" scores about -1 and deeper savings push the reward toward 0.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import OPERATOR_DEFAULT, DRXConfig, PowerModel, RadioConfig, build_action_grid
from .simulator import DRXSimulator, WindowResult
from .traffic import PROFILES, TrafficScenario

OBS_NAMES = (
    "log_pkt_count",  # log1p(packets) / log1p(200)
    "interarrival_cv",  # burstiness: 0 = periodic, ~1 = Poisson, >1 = bursty
    "burst_fraction",  # share of inter-arrivals under 10 ms
    "max_gap_frac",  # longest silence / window length
    "violation_rate",  # share of packets that missed their latency budget
    "awake_frac",  # share of the window spent awake
    "ewma_log_pkt_count",  # slow average of activity: "is this user busy lately?"
    "ewma_periodic",  # slow average of "was the last window periodic traffic?"
)

EWMA_ALPHA = 0.3  # weight of the newest window in the history features


def window_features(
    arrivals: np.ndarray, result: WindowResult, window_ms: float
) -> np.ndarray:
    n = arrivals.size
    if n >= 2:
        ia = np.diff(arrivals)
        cv = float(np.std(ia) / max(np.mean(ia), 1e-9))
        burst = float(np.mean(ia < 10.0))
        max_gap = float(max(ia.max(), arrivals[0] % window_ms, window_ms - arrivals[-1] % window_ms))
    else:
        cv, burst, max_gap = 0.0, 0.0, window_ms
    return np.array(
        [
            min(np.log1p(n) / np.log1p(200), 1.0),
            min(cv / 3.0, 1.0),
            burst,
            min(max_gap / window_ms, 1.0),
            result.violation_rate,
            min(result.awake_ms / window_ms, 1.0),
        ],
        dtype=np.float32,
    )


class DRXEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        window_ms: float = 1000.0,
        episode_windows: int = 60,
        lambda_latency: float = 2.0,
        profiles: tuple[str, ...] = tuple(PROFILES),
        mean_sojourn_windows: float = 8.0,
        power: PowerModel | None = None,
        radio: RadioConfig | None = None,
        actions: list[DRXConfig] | None = None,
        config_override: DRXConfig | None = None,
    ) -> None:
        super().__init__()
        self.window_ms = window_ms
        self.episode_windows = episode_windows
        self.lambda_latency = lambda_latency
        self.profiles = profiles
        self.mean_sojourn_windows = mean_sojourn_windows
        self.sim = DRXSimulator(power, radio)
        self.actions = actions or build_action_grid()
        for cfg in self.actions:
            cfg.validate(self.sim.radio)
        self.config_override = config_override  # evaluation hook for reference baselines
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(OBS_NAMES),), dtype=np.float32)
        self._norm = self.sim.power.pdcch_monitor * window_ms
        self._scenario: TrafficScenario | None = None
        self._ewma: np.ndarray | None = None
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.sim.reset()
        # Traffic gets its own RNG stream so that, for a given seed, every policy
        # faces exactly the same packet trace (fair, paired comparisons).
        traffic_rng = np.random.default_rng(self.np_random.integers(2**63))
        self._scenario = TrafficScenario(traffic_rng, self.profiles, self.mean_sojourn_windows)
        self._step = 0
        self._ewma = None
        obs, _, _ = self._advance(self.config_override or OPERATOR_DEFAULT)  # warm-up window
        return obs, {}

    def step(self, action: int):
        assert self._scenario is not None, "call reset() first"
        cfg = self.config_override or self.actions[int(action)]
        obs, result, profile = self._advance(cfg)
        energy_norm = result.energy / self._norm
        reward = -(energy_norm + self.lambda_latency * result.violation_rate)
        self._step += 1
        info = {
            "profile": profile,
            "config": cfg.label,
            "energy": result.energy,
            "energy_norm": energy_norm,
            "violation_rate": result.violation_rate,
            "latencies_ms": result.latencies_ms,
            "budgets_ms": result.budgets_ms,
        }
        return obs, float(reward), False, self._step >= self.episode_windows, info

    def _advance(self, cfg: DRXConfig):
        win = self._scenario.next_window(self.window_ms)
        result = self.sim.run_window(win.arrivals, win.tx_ms, win.latency_budget_ms, cfg, self.window_ms)
        f = window_features(win.arrivals, result, self.window_ms)
        # History features. A single window cannot distinguish a paused VoIP call
        # from an idle IoT device (both have zero packets); the slow averages can.
        periodic = float(win.arrivals.size >= 5 and f[1] * 3.0 < 0.5)
        if self._ewma is None:
            self._ewma = np.array([f[0], periodic])
        else:
            self._ewma = (1 - EWMA_ALPHA) * self._ewma + EWMA_ALPHA * np.array([f[0], periodic])
        obs = np.concatenate([f, self._ewma]).astype(np.float32)
        return obs, result, win.profile
