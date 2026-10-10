"""Policies: fixed configs, a hand-written heuristic, and a tabular Q-learning agent.

Every policy implements the same tiny interface (``Policy.act``), so the
evaluation harness treats learned and hand-written controllers identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DRXConfig


class Policy(Protocol):
    name: str

    def act(self, obs: np.ndarray) -> int: ...


def action_index(actions: list[DRXConfig], inactivity_ms: float, cycle_ms: float) -> int:
    for i, cfg in enumerate(actions):
        if cfg.inactivity_ms == inactivity_ms and cfg.cycle_ms == cycle_ms:
            return i
    raise KeyError(f"no action with inactivity={inactivity_ms}, cycle={cycle_ms}")


@dataclass
class FixedPolicy:
    action: int
    name: str

    def act(self, obs: np.ndarray) -> int:
        return self.action


class HeuristicPolicy:
    """Rule-of-thumb controller an engineer might write by hand (no learning).

    Each rule targets one traffic regime: sparse traffic gets a moderate cycle
    (a long cycle would hurt the next burst), periodic traffic gets a cycle
    shorter than a VoIP budget, bursts get a long inactivity timer so the UE
    stays awake through the burst, and dense random traffic gets short cycles.
    """

    name = "heuristic"

    def __init__(self, actions: list[DRXConfig]) -> None:
        self.sparse = action_index(actions, 50.0, 80.0)
        self.periodic = action_index(actions, 10.0, 40.0)
        self.bursty = action_index(actions, 50.0, 160.0)
        self.dense_random = action_index(actions, 10.0, 20.0)

    def act(self, obs: np.ndarray) -> int:
        log_n, cv, burst = obs[0], obs[1] * 3.0, obs[2]
        if log_n < 0.15:  # almost no traffic last window
            return self.sparse
        if cv < 0.5:  # regular, periodic traffic
            return self.periodic
        if burst > 0.5:  # packets arrive in tight bursts
            return self.bursty
        return self.dense_random


class Discretizer:
    """Maps the continuous observation to a single integer state id."""

    def __init__(self, bins: tuple[tuple[float, ...], ...] | None = None) -> None:
        # Cut points per feature (see env.OBS_NAMES). Chosen to separate the
        # traffic regimes that call for different DRX settings.
        self.bins = bins or (
            (0.05, 0.25, 0.5, 0.7),  # packet count
            (0.15, 0.45),  # inter-arrival CV
            (0.3, 0.7),  # burst fraction
            (),  # longest gap (observed, but not used for the table)
            (0.05,),  # any latency violations
            (),  # awake fraction (not used)
            (0.25, 0.55),  # recent activity (history)
            (0.5,),  # recently periodic (history)
        )
        self.sizes = tuple(len(b) + 1 for b in self.bins)
        self.n_states = int(np.prod(self.sizes))

    def __call__(self, obs: np.ndarray) -> int:
        idx = [int(np.searchsorted(cuts, obs[i], side="right")) for i, cuts in enumerate(self.bins)]
        return int(np.ravel_multi_index(idx, self.sizes))


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    Tabular on purpose: the learned table is fully inspectable, so we can show
    *which* DRX config the agent picks in *which* traffic regime, something a
    reviewer can sanity-check against wireless intuition.
    """

    name = "rl_q_learning"

    def __init__(
        self,
        n_actions: int,
        discretizer: Discretizer | None = None,
        alpha: float = 0.1,
        gamma: float = 0.5,
        seed: int = 0,
        fallback_action: int = 0,
    ) -> None:
        self.disc = discretizer or Discretizer()
        self.fallback_action = fallback_action
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.q = np.zeros((self.disc.n_states, n_actions))
        self.visits = np.zeros_like(self.q, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0

    def act(self, obs: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        s = self.disc(obs)
        tried = self.visits[s] > 0
        if not tried.any():  # unseen traffic regime: fall back to a safe default
            return self.fallback_action
        q = np.where(tried, self.q[s], -np.inf)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, obs, action, reward, next_obs, done) -> None:
        s, s2 = self.disc(obs), self.disc(next_obs)
        target = reward + (0.0 if done else self.gamma * self.q[s2].max())
        self.q[s, action] += self.alpha * (target - self.q[s, action])
        self.visits[s, action] += 1

    def train(self, env, episodes: int, seed: int = 0, eps_start=1.0, eps_end=0.05, log_every=0):
        """Train in-place; returns per-episode total reward (the learning curve)."""
        returns = []
        for ep in range(episodes):
            self.epsilon = eps_start + (eps_end - eps_start) * min(1.0, ep / (0.8 * episodes))
            obs, _ = env.reset(seed=seed + ep)
            done, total = False, 0.0
            while not done:
                a = self.act(obs)
                nxt, r, term, trunc, _ = env.step(a)
                done = term or trunc
                self.update(obs, a, r, nxt, done)
                obs, total = nxt, total + r
            returns.append(total)
            if log_every and (ep + 1) % log_every == 0:
                print(f"episode {ep + 1:5d}  eps={self.epsilon:.2f}  "
                      f"mean return (last {log_every}) = {np.mean(returns[-log_every:]):.2f}")
        self.epsilon = 0.0
        return np.asarray(returns)

    def save(self, path: str | Path) -> None:
        np.savez(path, q=self.q, visits=self.visits, alpha=self.alpha, gamma=self.gamma,
                 fallback_action=self.fallback_action)

    @classmethod
    def load(cls, path: str | Path, discretizer: Discretizer | None = None) -> QLearningAgent:
        data = np.load(path)
        agent = cls(data["q"].shape[1], discretizer, float(data["alpha"]), float(data["gamma"]),
                    fallback_action=int(data["fallback_action"]))
        if agent.q.shape != data["q"].shape:
            raise ValueError("saved Q-table does not match the discretizer")
        agent.q, agent.visits = data["q"], data["visits"]
        return agent
