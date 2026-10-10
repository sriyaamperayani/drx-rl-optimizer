"""Paired evaluation of policies on identical traffic traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from .agents import FixedPolicy, Policy
from .config import DRXConfig
from .env import DRXEnv


@dataclass
class ProfileStats:
    energy_norm: float
    violation_rate: float
    mean_latency_ms: float
    config_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class EvalResult:
    name: str
    mean_return: float
    energy_norm: float  # 1.0 is roughly "never sleep"
    violation_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    per_profile: dict[str, ProfileStats]
    saving_pct: float = float("nan")  # filled in relative to the DRX-off reference

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(
    policy: Policy,
    env_kwargs: dict,
    seeds: list[int],
    config_override: DRXConfig | None = None,
) -> EvalResult:
    """Run ``policy`` on one episode per seed. Same seeds => same traffic for every policy."""
    env = DRXEnv(**env_kwargs, config_override=config_override)
    returns, energies, lat_all, viol_all = [], [], [], []
    prof_energy = defaultdict(list)
    prof_lat = defaultdict(list)
    prof_viol = defaultdict(list)
    prof_cfg: dict[str, Counter] = defaultdict(Counter)

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            done = term or trunc
            total += r
            p = info["profile"]
            energies.append(info["energy_norm"])
            prof_energy[p].append(info["energy_norm"])
            prof_cfg[p][info["config"]] += 1
            lat, bud = info["latencies_ms"], info["budgets_ms"]
            lat_all.append(lat)
            viol_all.append(lat > bud)
            prof_lat[p].append(lat)
            prof_viol[p].append(lat > bud)
        returns.append(total)

    lat = np.concatenate(lat_all) if lat_all else np.empty(0)
    viol = np.concatenate(viol_all) if viol_all else np.empty(0, dtype=bool)

    def _mean(x):
        return float(np.mean(x)) if np.size(x) else 0.0

    per_profile = {
        p: ProfileStats(
            energy_norm=_mean(prof_energy[p]),
            violation_rate=_mean(np.concatenate(prof_viol[p])),
            mean_latency_ms=_mean(np.concatenate(prof_lat[p])),
            config_counts=dict(prof_cfg[p]),
        )
        for p in sorted(prof_energy)
    }
    return EvalResult(
        name=policy.name,
        mean_return=_mean(returns),
        energy_norm=_mean(energies),
        violation_rate=_mean(viol),
        mean_latency_ms=_mean(lat),
        p95_latency_ms=float(np.percentile(lat, 95)) if lat.size else 0.0,
        per_profile=per_profile,
    )


def evaluate_reference(config: DRXConfig, name: str, env_kwargs: dict, seeds: list[int]) -> EvalResult:
    """Evaluate a config that lives outside the action grid (e.g. DRX disabled)."""
    result = evaluate(FixedPolicy(0, name), env_kwargs, seeds, config_override=config)
    return result


def attach_savings(results: list[EvalResult], reference: EvalResult) -> None:
    for r in results:
        r.saving_pct = 100.0 * (1.0 - r.energy_norm / reference.energy_norm)
