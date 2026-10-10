import numpy as np
from gymnasium.utils.env_checker import check_env

from drxgym.agents import Discretizer, FixedPolicy, HeuristicPolicy, QLearningAgent
from drxgym.config import DRX_OFF, build_action_grid
from drxgym.env import DRXEnv
from drxgym.evaluation import attach_savings, evaluate, evaluate_reference


def test_passes_gymnasium_env_checker():
    check_env(DRXEnv(episode_windows=5), skip_render_check=True)


def test_episode_length_and_observation_bounds():
    env = DRXEnv(episode_windows=7)
    obs, _ = env.reset(seed=0)
    steps, done = 0, False
    while not done:
        assert env.observation_space.contains(obs)
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
        steps += 1
        assert r <= 0
    assert steps == 7


def test_same_seed_gives_identical_returns():
    env = DRXEnv(episode_windows=20)
    totals = []
    for _ in range(2):
        env.reset(seed=123)
        totals.append(sum(env.step(3)[1] for _ in range(20)))
    assert totals[0] == totals[1]


def test_drx_off_reference_energy_is_about_one():
    ref = evaluate_reference(DRX_OFF, "drx_off", dict(episode_windows=10), seeds=[0, 1])
    assert 1.0 <= ref.energy_norm < 1.2
    assert ref.violation_rate == 0.0


def test_any_drx_config_saves_energy_vs_drx_off():
    kw, seeds = dict(episode_windows=10), [0, 1]
    ref = evaluate_reference(DRX_OFF, "drx_off", kw, seeds)
    res = [evaluate(FixedPolicy(i, f"a{i}"), kw, seeds) for i in range(len(build_action_grid()))]
    attach_savings(res, ref)
    assert all(r.saving_pct > 0 for r in res)


def test_discretizer_covers_observation_space():
    d = Discretizer()
    rng = np.random.default_rng(0)
    ids = {d(rng.random(8).astype(np.float32)) for _ in range(5000)}
    assert max(ids) < d.n_states and min(ids) >= 0


def test_q_learning_improves_over_random_and_roundtrips(tmp_path):
    env = DRXEnv(episode_windows=30)
    agent = QLearningAgent(env.action_space.n, seed=0)
    returns = agent.train(env, episodes=150, seed=0)
    assert returns[-30:].mean() > returns[:30].mean()
    path = tmp_path / "q.npz"
    agent.save(path)
    loaded = QLearningAgent.load(path)
    np.testing.assert_array_equal(agent.q, loaded.q)
    assert loaded.fallback_action == agent.fallback_action


def test_heuristic_returns_valid_actions():
    env = DRXEnv(episode_windows=10)
    pol = HeuristicPolicy(env.actions)
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        a = pol.act(obs)
        assert env.action_space.contains(a)
        obs, *_ = env.step(a)


def test_history_features_separate_paused_call_from_idle_device():
    """Most silent windows during a VoIP call must not look like an idle IoT device.

    Not all: after a long silence (or at episode start) the two are genuinely
    indistinguishable from traffic statistics alone.
    """
    d = Discretizer()

    def idle_states(profile):
        env, states = DRXEnv(profiles=(profile,)), []
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            for _ in range(60):
                obs, *_ = env.step(0)
                if obs[0] == 0.0:  # empty window
                    states.append(d(obs))
        return states

    voip, iot = idle_states("voip"), set(idle_states("iot"))
    separable = sum(s not in iot for s in voip) / len(voip)
    assert separable > 0.7
