"""Retrain the agent with different seeds to check results are not a lucky draw.

Usage: python scripts/seed_robustness.py --lambda-latency 2 --seed 1
Appends one row per run to runs/seed_robustness.jsonl.
"""

import argparse
import json
from pathlib import Path

from drxgym.agents import QLearningAgent, action_index
from drxgym.config import DRX_OFF, OPERATOR_DEFAULT, build_action_grid
from drxgym.env import DRXEnv
from drxgym.evaluation import attach_savings, evaluate, evaluate_reference
from drxgym.experiment import EVAL_SEED_OFFSET, ExperimentConfig

p = argparse.ArgumentParser()
p.add_argument("--lambda-latency", type=float, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--beam-search-ms", type=float, default=4.0)
args = p.parse_args()

cfg = ExperimentConfig(lambda_latency=args.lambda_latency, beam_search_ms=args.beam_search_ms)
kw = cfg.env_kwargs()
actions = build_action_grid()
default = action_index(actions, OPERATOR_DEFAULT.inactivity_ms, OPERATOR_DEFAULT.cycle_ms)
agent = QLearningAgent(len(actions), seed=args.seed, fallback_action=default)
agent.train(DRXEnv(**kw), cfg.train_episodes, seed=args.seed * 100_000)

# Evaluation traces are the same held-out set used by the main experiment.
seeds = [EVAL_SEED_OFFSET + i for i in range(cfg.eval_episodes)]
ref = evaluate_reference(DRX_OFF, "drx_off", kw, seeds)
rl = evaluate(agent, kw, seeds)
attach_savings([rl], ref)
row = {"lambda": args.lambda_latency, "beam_search_ms": args.beam_search_ms, "seed": args.seed,
       "return": rl.mean_return, "saving_pct": rl.saving_pct, "violation_pct": 100 * rl.violation_rate}
out = Path("runs/seed_robustness.jsonl")
with out.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(row)
