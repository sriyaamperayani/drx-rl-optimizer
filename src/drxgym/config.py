"""Configuration objects: UE power model, radio parameters, DRX configs, action grid.

All power values are in *relative power units* (rpu), following the style of the
3GPP UE power-consumption model used in TR 38.840 (Rel-16 UE power saving study),
where deep sleep = 1 and every other state is expressed relative to it. Absolute
milliwatts are deliberately avoided: the goal is to compare policies, not to
predict a specific handset's battery life.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerModel:
    """Per-state power (rpu) and sleep transition overheads.

    Sleep states have a transition cost: entering and leaving deep sleep takes
    time and a fixed amount of energy, so deep sleep only pays off for long gaps.
    For any idle gap the simulator picks whichever feasible sleep state is
    cheapest, which mirrors how a real modem chooses its sleep depth.
    """

    deep_sleep: float = 1.0
    light_sleep: float = 20.0
    micro_sleep: float = 45.0
    pdcch_monitor: float = 100.0  # awake, listening for scheduling grants
    pdsch_rx: float = 300.0  # actively receiving data
    beam_search: float = 150.0  # mmWave beam sweep / alignment after waking

    deep_sleep_transition_ms: float = 20.0
    deep_sleep_transition_energy: float = 450.0
    light_sleep_transition_ms: float = 6.0
    light_sleep_transition_energy: float = 100.0

    def sleep_energy(self, gap_ms: float) -> float:
        """Energy (rpu x ms) to spend an idle gap of ``gap_ms`` in the cheapest sleep state."""
        if gap_ms <= 0:
            return 0.0
        options = [self.micro_sleep * gap_ms]
        if gap_ms >= self.light_sleep_transition_ms:
            options.append(
                self.light_sleep * (gap_ms - self.light_sleep_transition_ms)
                + self.light_sleep_transition_energy
            )
        if gap_ms >= self.deep_sleep_transition_ms:
            options.append(
                self.deep_sleep * (gap_ms - self.deep_sleep_transition_ms)
                + self.deep_sleep_transition_energy
            )
        return min(options)


@dataclass(frozen=True)
class RadioConfig:
    """Radio-specific timing.

    ``beam_search_ms`` models the extra alignment time a mmWave (FR2) UE needs
    after every sleep period before it can decode the control channel, which
    is the core idea behind *directional* DRX. Set it to 0 to model FR1 (sub-6 GHz).
    """

    name: str = "FR2-mmWave"
    beam_search_ms: float = 4.0


@dataclass(frozen=True)
class DRXConfig:
    """Connected-mode DRX parameters (simplified to a single long cycle)."""

    inactivity_ms: float
    cycle_ms: float
    on_duration_ms: float = 8.0

    def validate(self, radio: RadioConfig) -> None:
        if self.inactivity_ms < 0:
            raise ValueError("inactivity_ms must be non-negative")
        if self.on_duration_ms <= 0:
            raise ValueError("on_duration_ms must be positive")
        if self.cycle_ms < self.on_duration_ms + radio.beam_search_ms:
            raise ValueError(
                "cycle_ms must be at least on_duration_ms + beam_search_ms "
                f"({self.on_duration_ms + radio.beam_search_ms} ms)"
            )

    @property
    def always_on(self) -> bool:
        """True for the 'DRX disabled' reference: the UE never sleeps."""
        return math.isinf(self.inactivity_ms)

    @property
    def label(self) -> str:
        if self.always_on:
            return "drx_off"
        return f"inact={self.inactivity_ms:g}ms/cycle={self.cycle_ms:g}ms"


# Reference config that never sleeps: the "DRX disabled" baseline used to
# compute energy savings.
DRX_OFF = DRXConfig(inactivity_ms=math.inf, cycle_ms=math.inf)

# A typical operator default, used as the "status quo" baseline.
OPERATOR_DEFAULT = DRXConfig(inactivity_ms=100.0, cycle_ms=160.0)

INACTIVITY_CHOICES_MS = (10.0, 50.0, 100.0, 200.0)
CYCLE_CHOICES_MS = (20.0, 40.0, 80.0, 160.0, 320.0)


def build_action_grid(
    inactivity_choices: tuple[float, ...] = INACTIVITY_CHOICES_MS,
    cycle_choices: tuple[float, ...] = CYCLE_CHOICES_MS,
    on_duration_ms: float = 8.0,
) -> list[DRXConfig]:
    """Discrete action space: every (inactivity timer, cycle length) combination."""
    return [
        DRXConfig(inactivity_ms=i, cycle_ms=c, on_duration_ms=on_duration_ms)
        for i, c in itertools.product(inactivity_choices, cycle_choices)
    ]
