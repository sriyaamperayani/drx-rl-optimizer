"""Event-driven simulator of a UE's connected-mode DRX behaviour.

Timeline model (all times in ms):

* **Awake**: after receiving data the UE stays awake for ``inactivity_ms``,
  monitoring the control channel (PDCCH). New packets are served immediately
  (FIFO, one at a time) and each one restarts the inactivity timer.
* **DRX**: when the timer expires the UE enters a DRX cycle anchored at that
  instant. Every ``cycle_ms`` it wakes up, performs a beam search of
  ``beam_search_ms`` (mmWave only), then monitors for ``on_duration_ms``.
  Between on-durations it sleeps in the cheapest feasible sleep state.
* A packet that arrives while the UE sleeps is buffered at the base station
  and delivered at the first on-duration that is still open, after the beam
  search completes. That buffering delay is the latency cost of saving energy.

The simulator is event-driven (it jumps between packet arrivals and cycle
boundaries) instead of stepping every millisecond, which makes RL training
roughly two orders of magnitude faster.

Invariant (unit-tested): the time accounted across all states in a window
equals the window length exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import DRXConfig, PowerModel, RadioConfig


@dataclass
class WindowResult:
    energy: float  # rpu x ms
    accounted_ms: float
    awake_ms: float
    sleep_ms: float
    beam_ms: float
    on_durations: int
    latencies_ms: np.ndarray
    budgets_ms: np.ndarray

    @property
    def n_packets(self) -> int:
        return int(self.latencies_ms.size)

    @property
    def violation_rate(self) -> float:
        if self.latencies_ms.size == 0:
            return 0.0
        return float(np.mean(self.latencies_ms > self.budgets_ms))


@dataclass
class _Acc:
    energy: float = 0.0
    awake: float = 0.0
    sleep: float = 0.0
    beam: float = 0.0
    on_durations: int = 0
    latencies: list[float] = field(default_factory=list)
    budgets: list[float] = field(default_factory=list)


class DRXSimulator:
    def __init__(self, power: PowerModel | None = None, radio: RadioConfig | None = None) -> None:
        self.power = power or PowerModel()
        self.radio = radio or RadioConfig()
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.active_until = 0.0  # start already in DRX
        self.busy_until = 0.0
        self.drx_anchor = 0.0
        self._cfg: DRXConfig | None = None
        self._carry_t = np.empty(0)
        self._carry_tx = np.empty(0)
        self._carry_b = np.empty(0)

    # ------------------------------------------------------------------ public
    def run_window(
        self,
        arrivals: np.ndarray,
        tx_ms: np.ndarray,
        budgets_ms: np.ndarray | float,
        cfg: DRXConfig,
        window_ms: float,
    ) -> WindowResult:
        """Simulate one decision window under ``cfg`` and advance the clock."""
        cfg.validate(self.radio)
        w0, w1 = self.t, self.t + window_ms
        budgets = np.broadcast_to(np.asarray(budgets_ms, dtype=float), np.shape(arrivals))
        if arrivals.size and (arrivals.min() < w0 or arrivals.max() >= w1):
            raise ValueError("arrivals must lie inside the current window")

        # Reconfiguration while sleeping restarts the cycle at the window boundary.
        if self._cfg is not None and cfg != self._cfg and self.active_until < w0:
            self.drx_anchor = w0
        self._cfg = cfg
        if cfg.always_on:
            self.active_until = math.inf

        t_all = np.concatenate([self._carry_t, arrivals])
        tx_all = np.concatenate([self._carry_tx, tx_ms])
        b_all = np.concatenate([self._carry_b, budgets])
        self._carry_t = self._carry_tx = self._carry_b = np.empty(0)

        acc = _Acc()
        cursor = w0  # energy has been accounted up to here
        B = self.radio.beam_search_ms

        for i, (a, d, b) in enumerate(zip(t_all, tx_all, b_all, strict=True)):
            if a <= self.active_until:
                start = max(a, self.busy_until)
            else:
                s = self._serving_on_duration_start(a, cfg)
                if s + B >= w1:
                    # Next opportunity is in a future window: carry the backlog.
                    self._carry_t, self._carry_tx, self._carry_b = t_all[i:], tx_all[i:], b_all[i:]
                    break
                cursor = self._account_awake(acc, cursor, self.active_until)
                self._account_drx(acc, cursor, s, cfg)
                # The serving on-duration may have started in the previous
                # window; only account the part after ``cursor`` (no double count).
                self._add(acc, "beam", self.power.beam_search, max(s, cursor), s + B)
                if s >= cursor:
                    acc.on_durations += 1
                cursor = max(cursor, s + B)
                start = max(a, s + B, self.busy_until)

            end = start + d
            acc.energy += d * (self.power.pdsch_rx - self.power.pdcch_monitor)
            acc.latencies.append(end - a)
            acc.budgets.append(b)
            self.busy_until = end
            self.active_until = end + cfg.inactivity_ms
            self.drx_anchor = self.active_until

        # Close the window.
        if self.active_until >= w1:
            self._account_awake(acc, cursor, w1)
        else:
            cursor = self._account_awake(acc, cursor, self.active_until)
            self._account_drx(acc, cursor, w1, cfg)
        self.t = w1

        return WindowResult(
            energy=acc.energy,
            accounted_ms=acc.awake + acc.sleep + acc.beam,
            awake_ms=acc.awake,
            sleep_ms=acc.sleep,
            beam_ms=acc.beam,
            on_durations=acc.on_durations,
            latencies_ms=np.asarray(acc.latencies),
            budgets_ms=np.asarray(acc.budgets),
        )

    # ----------------------------------------------------------------- helpers
    def _serving_on_duration_start(self, a: float, cfg: DRXConfig) -> float:
        """Start time of the first on-duration that is still open at time ``a``."""
        B, D, C = self.radio.beam_search_ms, cfg.on_duration_ms, cfg.cycle_ms
        k = max(1, math.floor((a - self.drx_anchor - B - D) / C) + 1)
        return self.drx_anchor + k * C

    def _add(self, acc: _Acc, kind: str, power: float, t0: float, t1: float) -> float:
        dur = max(0.0, t1 - t0)
        acc.energy += power * dur
        setattr(acc, kind, getattr(acc, kind) + dur)
        return dur

    def _account_awake(self, acc: _Acc, t0: float, t1: float) -> float:
        self._add(acc, "awake", self.power.pdcch_monitor, t0, t1)
        return max(t0, t1)

    def _account_drx(self, acc: _Acc, t_from: float, t_to: float, cfg: DRXConfig) -> None:
        """Account sleep / beam-search / on-duration segments of the DRX cycle in [t_from, t_to)."""
        if t_to <= t_from:
            return
        B, D, C = self.radio.beam_search_ms, cfg.on_duration_ms, cfg.cycle_ms
        anchor = self.drx_anchor
        k = max(0, math.floor((t_from - anchor) / C) - 1)
        while True:
            sleep_start = anchor if k == 0 else anchor + k * C + B + D
            if sleep_start >= t_to:
                break
            on_start = anchor + (k + 1) * C
            gap = on_start - sleep_start
            lo, hi = max(sleep_start, t_from), min(on_start, t_to)
            if hi > lo:  # pro-rate the (transition-aware) sleep energy
                acc.energy += self.power.sleep_energy(gap) * (hi - lo) / gap
                acc.sleep += hi - lo
            if t_from <= on_start < t_to:
                acc.on_durations += 1
            beam_end, on_end = on_start + B, on_start + B + D
            self._add(acc, "beam", self.power.beam_search, max(on_start, t_from), min(beam_end, t_to))
            self._add(acc, "awake", self.power.pdcch_monitor, max(beam_end, t_from), min(on_end, t_to))
            k += 1
