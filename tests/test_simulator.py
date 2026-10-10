import numpy as np
import pytest

from drxgym.config import DRX_OFF, DRXConfig, PowerModel, RadioConfig, build_action_grid
from drxgym.simulator import DRXSimulator
from drxgym.traffic import PROFILES, TrafficScenario

W = 1000.0
EMPTY = np.empty(0)


def run(sim, arrivals, cfg, budget=100.0):
    arrivals = np.asarray(arrivals, dtype=float)
    return sim.run_window(arrivals, np.ones_like(arrivals), budget, cfg, W)


def test_always_on_with_no_traffic_costs_pdcch_power():
    sim = DRXSimulator()
    res = run(sim, [], DRX_OFF)
    assert res.energy == pytest.approx(PowerModel().pdcch_monitor * W)
    assert res.awake_ms == pytest.approx(W)


def test_always_on_serves_immediately():
    res = run(DRXSimulator(), [100.0, 500.0], DRX_OFF)
    np.testing.assert_allclose(res.latencies_ms, [1.0, 1.0])  # just the transmission time


def test_drx_with_no_traffic_is_mostly_asleep():
    cfg = DRXConfig(inactivity_ms=10, cycle_ms=160)
    res = run(DRXSimulator(), [], cfg)
    assert res.sleep_ms > 0.9 * W
    assert res.energy < 0.2 * PowerModel().pdcch_monitor * W


def test_sleeping_packet_waits_for_next_on_duration_plus_beam_search():
    radio = RadioConfig(beam_search_ms=4.0)
    cfg = DRXConfig(inactivity_ms=10, cycle_ms=100)
    sim = DRXSimulator(radio=radio)
    res = run(sim, [150.0], cfg)  # UE asleep since t=0; on-durations at 100, 200, ...
    # Arrives after the on-duration at 100 closed (100+4+8=112), so served at 200 + 4 (beam) + 1 (tx).
    np.testing.assert_allclose(res.latencies_ms, [55.0])


def test_packet_during_on_duration_is_served_without_extra_wait():
    cfg = DRXConfig(inactivity_ms=10, cycle_ms=100)
    res = run(DRXSimulator(radio=RadioConfig(beam_search_ms=4.0)), [106.0], cfg)
    np.testing.assert_allclose(res.latencies_ms, [1.0])


def test_fifo_queueing_while_awake():
    cfg = DRXConfig(inactivity_ms=100, cycle_ms=160)
    res = run(DRXSimulator(radio=RadioConfig(beam_search_ms=0.0)), [160.0, 160.0, 160.0], cfg)
    np.testing.assert_allclose(res.latencies_ms, [1.0, 2.0, 3.0])


def test_backlog_carries_into_next_window():
    cfg = DRXConfig(inactivity_ms=10, cycle_ms=320)
    sim = DRXSimulator()
    first = run(sim, [990.0], cfg)  # next on-duration is at 1280, in the next window
    assert first.n_packets == 0
    second = sim.run_window(EMPTY, EMPTY, 100.0, cfg, W)
    assert second.n_packets == 1
    assert second.latencies_ms[0] == pytest.approx(1280 + 4 + 1 - 990)


@pytest.mark.parametrize("cfg", build_action_grid() + [DRX_OFF])
@pytest.mark.parametrize("profile", list(PROFILES))
def test_time_is_conserved_every_window(cfg, profile):
    """Core invariant: every millisecond is accounted to exactly one state."""
    rng = np.random.default_rng(0)
    scenario = TrafficScenario(rng, (profile,))
    sim = DRXSimulator()
    for _ in range(10):
        win = scenario.next_window(W)
        res = sim.run_window(win.arrivals, win.tx_ms, win.latency_budget_ms, cfg, W)
        assert res.accounted_ms == pytest.approx(W, abs=1e-6)
        assert np.all(res.latencies_ms > 0)


def test_reconfiguration_mid_episode_keeps_invariant():
    rng = np.random.default_rng(1)
    scenario = TrafficScenario(rng)
    sim = DRXSimulator()
    grid = build_action_grid()
    for k in range(40):
        win = scenario.next_window(W)
        res = sim.run_window(win.arrivals, win.tx_ms, win.latency_budget_ms, grid[k % len(grid)], W)
        assert res.accounted_ms == pytest.approx(W, abs=1e-6)


def test_more_sleep_means_less_energy_but_more_latency():
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    short, long_ = DRXConfig(10, 20), DRXConfig(10, 320)
    out = {}
    for cfg, rng in ((short, rng_a), (long_, rng_b)):
        sc, sim, e, lat = TrafficScenario(rng, ("web",)), DRXSimulator(), 0.0, []
        for _ in range(30):
            w = sc.next_window(W)
            r = sim.run_window(w.arrivals, w.tx_ms, w.latency_budget_ms, cfg, W)
            e += r.energy
            lat.extend(r.latencies_ms)
        out[cfg] = (e, np.mean(lat))
    assert out[long_][0] < out[short][0]
    assert out[long_][1] > out[short][1]


def test_rejects_arrivals_outside_window():
    with pytest.raises(ValueError):
        run(DRXSimulator(), [1500.0], DRX_OFF)


def test_regression_on_duration_straddling_window_boundary_not_double_counted():
    """Bug found by the conservation test: a packet served in an on-duration that
    began in the previous window used to have its beam search counted twice."""
    radio = RadioConfig(beam_search_ms=4.0)
    cfg = DRXConfig(inactivity_ms=10, cycle_ms=100, on_duration_ms=8)
    sim = DRXSimulator(radio=radio)
    sim.drx_anchor = -3.0  # on-durations at 97, 197, ..., 997: the last straddles t=1000
    run(sim, [], cfg)
    second = sim.run_window(np.array([1002.0]), np.array([1.0]), 100.0, cfg, W)
    assert second.accounted_ms == pytest.approx(W, abs=1e-6)
    np.testing.assert_allclose(second.latencies_ms, [1.0])
