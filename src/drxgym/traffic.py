"""Downlink traffic models and a Markov-switching traffic scenario.

Each source is *stateful*: bursts that straddle a decision-window boundary
continue correctly into the next window. Times are in milliseconds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class TrafficSource(ABC):
    """Base class. Subclasses implement ``_extend`` to generate the next chunk of events."""

    name: str = "base"
    latency_budget_ms: float = 100.0

    def __init__(self, rng: np.random.Generator, start_ms: float = 0.0) -> None:
        self.rng = rng
        self._horizon = start_ms  # events have been generated up to here
        self._times: list[float] = []
        self._tx: list[float] = []

    @abstractmethod
    def _extend(self) -> None:
        """Append future events to ``_times``/``_tx`` and advance ``_horizon``."""

    def generate(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (arrival_times, tx_durations) for packets arriving in [t0, t1)."""
        while self._horizon < t1:
            self._extend()
        times = np.asarray(self._times, dtype=float)
        tx = np.asarray(self._tx, dtype=float)
        mask = times < t1
        keep = (times >= t0) & mask
        out_t, out_tx = times[keep], tx[keep]
        self._times = times[~mask].tolist()
        self._tx = tx[~mask].tolist()
        order = np.argsort(out_t, kind="stable")
        return out_t[order], out_tx[order]


class PoissonSource(TrafficSource):
    """Memoryless packet arrivals, e.g. IoT telemetry or online gaming."""

    def __init__(self, rng, start_ms=0.0, *, name, rate_per_s, tx_ms, latency_budget_ms):
        super().__init__(rng, start_ms)
        self.name = name
        self.rate_per_ms = rate_per_s / 1000.0
        self.tx_ms = tx_ms
        self.latency_budget_ms = latency_budget_ms

    def _extend(self) -> None:
        self._horizon += self.rng.exponential(1.0 / self.rate_per_ms)
        self._times.append(self._horizon)
        self._tx.append(self.tx_ms)


class VoIPSource(TrafficSource):
    """Talk spurts with one packet every ``period_ms``, separated by silences."""

    name = "voip"
    latency_budget_ms = 50.0

    def __init__(self, rng, start_ms=0.0, period_ms=20.0, mean_on_ms=2000.0, mean_off_ms=1500.0):
        super().__init__(rng, start_ms)
        self.period_ms = period_ms
        self.mean_on_ms = mean_on_ms
        self.mean_off_ms = mean_off_ms

    def _extend(self) -> None:
        on = self.rng.exponential(self.mean_on_ms)
        n = max(1, int(on // self.period_ms))
        base = self._horizon + np.arange(n) * self.period_ms
        jitter = self.rng.uniform(0.0, 2.0, size=n)
        self._times.extend((base + jitter).tolist())
        self._tx.extend([1.0] * n)
        self._horizon += n * self.period_ms + self.rng.exponential(self.mean_off_ms)


class WebSource(TrafficSource):
    """Page loads: a burst of packets after an exponential reading time."""

    name = "web"
    latency_budget_ms = 150.0

    def __init__(self, rng, start_ms=0.0, mean_read_ms=4000.0, mean_burst_pkts=25, mean_gap_ms=4.0):
        super().__init__(rng, start_ms)
        self.mean_read_ms = mean_read_ms
        self.p = 1.0 / mean_burst_pkts
        self.mean_gap_ms = mean_gap_ms

    def _extend(self) -> None:
        start = self._horizon + self.rng.exponential(self.mean_read_ms)
        n = int(self.rng.geometric(self.p))
        times = start + np.cumsum(self.rng.exponential(self.mean_gap_ms, size=n))
        self._times.extend(times.tolist())
        self._tx.extend([1.0] * n)
        self._horizon = float(times[-1])


class VideoSource(TrafficSource):
    """Adaptive-bitrate streaming: a large chunk roughly every ``period_ms``."""

    name = "video"
    latency_budget_ms = 400.0

    def __init__(self, rng, start_ms=0.0, period_ms=2000.0, chunk_pkts=60):
        super().__init__(rng, start_ms)
        self.period_ms = period_ms
        self.chunk_pkts = chunk_pkts

    def _extend(self) -> None:
        start = self._horizon + self.period_ms * self.rng.uniform(0.9, 1.1)
        times = start + np.arange(self.chunk_pkts, dtype=float)
        self._times.extend(times.tolist())
        self._tx.extend([1.0] * self.chunk_pkts)
        self._horizon = start


def _iot(rng, t0):
    return PoissonSource(rng, t0, name="iot", rate_per_s=0.2, tx_ms=1.0, latency_budget_ms=2000.0)


def _gaming(rng, t0):
    return PoissonSource(rng, t0, name="gaming", rate_per_s=30.0, tx_ms=1.0, latency_budget_ms=40.0)


SourceFactory = Callable[[np.random.Generator, float], TrafficSource]

PROFILES: dict[str, SourceFactory] = {
    "iot": _iot,
    "voip": lambda rng, t0: VoIPSource(rng, t0),
    "web": lambda rng, t0: WebSource(rng, t0),
    "video": lambda rng, t0: VideoSource(rng, t0),
    "gaming": _gaming,
}


@dataclass
class TrafficWindow:
    arrivals: np.ndarray
    tx_ms: np.ndarray
    profile: str
    latency_budget_ms: float


class TrafficScenario:
    """The user's activity switches between profiles as a Markov chain.

    At each window boundary the active profile changes with probability
    ``1 / mean_sojourn_windows`` to a uniformly chosen different profile.
    This non-stationarity is what makes a *learned, adaptive* DRX policy useful:
    no single fixed config is good for every profile.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        profiles: tuple[str, ...] = tuple(PROFILES),
        mean_sojourn_windows: float = 8.0,
    ) -> None:
        unknown = set(profiles) - set(PROFILES)
        if unknown:
            raise ValueError(f"unknown traffic profiles: {sorted(unknown)}")
        self.rng = rng
        self.profiles = profiles
        self.switch_prob = 1.0 / mean_sojourn_windows
        self.t = 0.0
        self.profile = str(rng.choice(profiles))
        self.source = PROFILES[self.profile](rng, 0.0)

    def next_window(self, window_ms: float) -> TrafficWindow:
        t0, t1 = self.t, self.t + window_ms
        if len(self.profiles) > 1 and self.rng.random() < self.switch_prob:
            choices = [p for p in self.profiles if p != self.profile]
            self.profile = str(self.rng.choice(choices))
            self.source = PROFILES[self.profile](self.rng, t0)
        arrivals, tx = self.source.generate(t0, t1)
        self.t = t1
        return TrafficWindow(arrivals, tx, self.profile, self.source.latency_budget_ms)
