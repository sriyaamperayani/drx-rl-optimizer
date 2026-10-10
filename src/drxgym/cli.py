"""Command-line interface.

    drxgym experiment --out runs/fr2          # train + evaluate + figures
    drxgym experiment --beam-search-ms 0      # FR1 (sub-6 GHz) ablation
    drxgym inspect runs/fr2/q_table.npz       # print the learned policy table
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .agents import QLearningAgent
from .config import build_action_grid
from .experiment import ExperimentConfig, run_experiment


def _experiment(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        window_ms=args.window_ms,
        episode_windows=args.episode_windows,
        lambda_latency=args.lambda_latency,
        beam_search_ms=args.beam_search_ms,
        seed=args.seed,
    )
    run_experiment(cfg, args.out, verbose=not args.quiet)


def _inspect(args: argparse.Namespace) -> None:
    agent = QLearningAgent.load(args.q_table)
    actions = build_action_grid()
    visited = np.flatnonzero(agent.visits.sum(axis=1) >= args.min_visits)
    print(f"{visited.size} traffic states visited at least {args.min_visits} times\n")
    for s in visited:
        tried = agent.visits[s] > 0
        best = int(np.argmax(np.where(tried, agent.q[s], -np.inf)))
        idx = np.unravel_index(s, agent.disc.sizes)
        print(f"state {s:4d} bins={tuple(int(i) for i in idx)}  visits={agent.visits[s].sum():6d}  "
              f"-> {actions[best].label}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drxgym", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("experiment", help="train the agent and benchmark it against baselines")
    e.add_argument("--out", default="runs/latest")
    e.add_argument("--train-episodes", type=int, default=5000)
    e.add_argument("--eval-episodes", type=int, default=200)
    e.add_argument("--window-ms", type=float, default=1000.0)
    e.add_argument("--episode-windows", type=int, default=60)
    e.add_argument("--lambda-latency", type=float, default=2.0,
                   help="weight of latency-budget violations in the reward")
    e.add_argument("--beam-search-ms", type=float, default=4.0,
                   help="mmWave beam alignment time after each wake-up (0 = FR1)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=_experiment)

    i = sub.add_parser("inspect", help="print the learned policy")
    i.add_argument("q_table")
    i.add_argument("--min-visits", type=int, default=50)
    i.set_defaults(func=_inspect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
