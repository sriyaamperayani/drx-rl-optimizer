"""End-to-end: the CLI trains, evaluates, and writes every report artifact."""

import json

from drxgym.cli import main


def test_experiment_and_inspect_end_to_end(tmp_path, capsys):
    out = tmp_path / "run"
    code = main(["experiment", "--out", str(out), "--train-episodes", "20",
                 "--eval-episodes", "3", "--episode-windows", "10", "--quiet"])
    assert code == 0
    for name in ("q_table.npz", "training_returns.npy", "results.json", "results.md",
                 "learning_curve.png", "pareto.png", "policy_map.png", "per_profile_savings.png"):
        assert (out / name).stat().st_size > 0, name

    summary = json.loads((out / "results.json").read_text())
    names = [r["name"] for r in summary["headline"]]
    assert names == ["operator_default", "best_static", "heuristic", "rl_q_learning"]
    assert len(summary["fixed"]) == 20
    assert "| rl_q_learning |" in (out / "results.md").read_text()

    assert main(["inspect", str(out / "q_table.npz"), "--min-visits", "1"]) == 0
    assert "traffic states visited" in capsys.readouterr().out
