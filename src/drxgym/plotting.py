"""Figures for the experiment report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluation import EvalResult  # noqa: E402

HIGHLIGHT = {
    "operator_default": ("#888888", "s"),
    "best_static": ("#1f77b4", "D"),
    "heuristic": ("#ff7f0e", "^"),
    "rl_q_learning": ("#d62728", "*"),
}


def learning_curve(returns: np.ndarray, path: Path, window: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(returns, alpha=0.25, color="#d62728", lw=0.8, label="episode return")
    if returns.size >= window:
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, returns.size), smooth, color="#d62728", lw=2,
                label=f"{window}-episode moving average")
    ax.set_xlabel("training episode")
    ax.set_ylabel("return (higher is better)")
    ax.set_title("Q-learning training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pareto(fixed: list[EvalResult], highlighted: list[EvalResult], path: Path) -> None:
    """Energy saving vs QoS violations: up and to the left is better."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter([100 * r.violation_rate for r in fixed], [r.saving_pct for r in fixed],
               color="#cccccc", edgecolor="#999999", s=40, label="fixed configs (all 20)")
    for r in highlighted:
        color, marker = HIGHLIGHT.get(r.name, ("black", "o"))
        ax.scatter(100 * r.violation_rate, r.saving_pct, color=color, marker=marker,
                   s=220 if marker == "*" else 90, edgecolor="black", zorder=5, label=r.name)
    ax.set_xlabel("packets over latency budget (%)  [lower is better]")
    ax.set_ylabel("energy saving vs DRX disabled (%)  [higher is better]")
    ax.set_title("Energy / QoS trade-off on held-out traffic")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _config_sort_key(label: str) -> tuple[float, float]:
    """Order 'inact=10ms/cycle=40ms' labels numerically, not alphabetically."""
    try:
        inact, cycle = label.split("/")
        return float(inact.split("=")[1].rstrip("ms")), float(cycle.split("=")[1].rstrip("ms"))
    except (ValueError, IndexError):
        return (float("inf"), float("inf"))


def policy_map(result: EvalResult, path: Path, min_share: float = 0.05) -> None:
    """Which DRX config the agent picks under each traffic profile."""
    profiles = list(result.per_profile)
    all_configs = sorted({c for s in result.per_profile.values() for c in s.config_counts},
                         key=_config_sort_key)
    full = np.zeros((len(profiles), len(all_configs)))
    for i, p in enumerate(profiles):
        counts = result.per_profile[p].config_counts
        total = sum(counts.values()) or 1
        for j, c in enumerate(all_configs):
            full[i, j] = counts.get(c, 0) / total
    keep = full.max(axis=0) >= min_share  # hide configs the agent essentially never uses
    m = full[:, keep]
    configs = [c for c, k in zip(all_configs, keep, strict=True) if k]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(configs) + 2), 0.6 * len(profiles) + 2))
    im = ax.imshow(m, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(configs)), [c.replace("/", "\n") for c in configs], fontsize=7)
    ax.set_yticks(range(len(profiles)), profiles)
    for i in range(len(profiles)):
        for j in range(len(configs)):
            if m[i, j] >= 0.05:
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="share of windows")
    ax.set_title(f"Learned policy: DRX config chosen per traffic profile\n"
                 f"(configs used in at least {min_share:.0%} of some profile's windows)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_profile_savings(results: list[EvalResult], reference: EvalResult, path: Path) -> None:
    profiles = list(reference.per_profile)
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, r in enumerate(results):
        vals = [100 * (1 - r.per_profile[p].energy_norm / reference.per_profile[p].energy_norm)
                if p in r.per_profile else 0 for p in profiles]
        color, _ = HIGHLIGHT.get(r.name, ("black", "o"))
        ax.bar(np.arange(len(profiles)) + (k - (len(results) - 1) / 2) * width, vals, width,
               label=r.name, color=color)
    ax.set_xticks(range(len(profiles)), profiles)
    ax.set_ylabel("energy saving vs DRX disabled (%)")
    ax.set_title("Energy saving by traffic profile")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
