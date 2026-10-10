"""End-to-end experiment: train, evaluate against baselines, write report artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import plotting
from .agents import FixedPolicy, HeuristicPolicy, QLearningAgent, action_index
from .config import DRX_OFF, OPERATOR_DEFAULT, RadioConfig, build_action_grid
from .env import DRXEnv
from .evaluation import EvalResult, attach_savings, evaluate, evaluate_reference

TRAIN_SEED_OFFSET = 0
EVAL_SEED_OFFSET = 1_000_000  # evaluation traces never overlap training traces


@dataclass
class ExperimentConfig:
    train_episodes: int = 5000
    eval_episodes: int = 200
    window_ms: float = 1000.0
    episode_windows: int = 60
    lambda_latency: float = 2.0
    beam_search_ms: float = 4.0
    seed: int = 0

    def env_kwargs(self) -> dict:
        return dict(
            window_ms=self.window_ms,
            episode_windows=self.episode_windows,
            lambda_latency=self.lambda_latency,
            radio=RadioConfig(name="FR2" if self.beam_search_ms else "FR1",
                              beam_search_ms=self.beam_search_ms),
        )


def run_experiment(cfg: ExperimentConfig, out_dir: str | Path, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    actions = build_action_grid()
    env_kwargs = cfg.env_kwargs()
    default_idx = action_index(actions, OPERATOR_DEFAULT.inactivity_ms, OPERATOR_DEFAULT.cycle_ms)

    # 1. Train
    t0 = time.time()
    agent = QLearningAgent(len(actions), seed=cfg.seed, fallback_action=default_idx)
    returns = agent.train(DRXEnv(**env_kwargs), cfg.train_episodes,
                          seed=cfg.seed + TRAIN_SEED_OFFSET,
                          log_every=max(1, cfg.train_episodes // 10) if verbose else 0)
    train_s = time.time() - t0
    agent.save(out / "q_table.npz")
    np.save(out / "training_returns.npy", returns)

    # 2. Evaluate on held-out traffic
    seeds = [cfg.seed + EVAL_SEED_OFFSET + i for i in range(cfg.eval_episodes)]
    reference = evaluate_reference(DRX_OFF, "drx_off", env_kwargs, seeds)
    fixed = [evaluate(FixedPolicy(i, a.label), env_kwargs, seeds) for i, a in enumerate(actions)]
    attach_savings(fixed, reference)
    # NOTE: "best_static" is tuned on the evaluation traces themselves, which
    # gives that baseline an unfair advantage. RL still has to beat it.
    best = max(fixed, key=lambda r: r.mean_return)
    best_static = EvalResult(**{**best.__dict__, "name": "best_static"})
    default = EvalResult(**{**fixed[default_idx].__dict__, "name": "operator_default"})
    heuristic = evaluate(HeuristicPolicy(actions), env_kwargs, seeds)
    rl = evaluate(agent, env_kwargs, seeds)
    attach_savings([heuristic, rl, reference], reference)
    headline = [default, best_static, heuristic, rl]

    # 3. Artifacts
    plotting.learning_curve(returns, out / "learning_curve.png")
    plotting.pareto(fixed, headline, out / "pareto.png")
    plotting.policy_map(rl, out / "policy_map.png")
    plotting.per_profile_savings([default, best_static, rl], reference, out / "per_profile_savings.png")

    summary = {
        "config": cfg.__dict__,
        "train_seconds": round(train_s, 1),
        "best_static_config": best.name,
        "reference": reference.to_dict(),
        "headline": [r.to_dict() for r in headline],
        "fixed": [r.to_dict() for r in fixed],
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    (out / "results.md").write_text(_markdown(cfg, headline, best.name, train_s))
    if verbose:
        print((out / "results.md").read_text())
    return summary


def _markdown(cfg: ExperimentConfig, rows: list[EvalResult], best_name: str, train_s: float) -> str:
    lines = [
        "# Results",
        "",
        f"Held-out evaluation: {cfg.eval_episodes} episodes x {cfg.episode_windows} windows "
        f"of {cfg.window_ms:g} ms, beam search {cfg.beam_search_ms:g} ms, "
        f"lambda = {cfg.lambda_latency:g}. Training: {cfg.train_episodes} episodes ({train_s:.0f} s).",
        "",
        "| Policy | Return | Energy saving | Packets over budget | Mean latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {r.mean_return:.2f} | {r.saving_pct:.1f}% | {100 * r.violation_rate:.2f}% "
            f"| {r.mean_latency_ms:.1f} ms | {r.p95_latency_ms:.1f} ms |"
        )
    lines += ["", f"best_static = `{best_name}` (chosen with hindsight on the evaluation traces).", ""]
    return "\n".join(lines)
