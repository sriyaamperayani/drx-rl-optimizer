import numpy as np
import pytest

from drxgym.traffic import PROFILES, TrafficScenario


@pytest.mark.parametrize("profile", list(PROFILES))
def test_windows_are_sorted_and_in_range(profile):
    sc = TrafficScenario(np.random.default_rng(0), (profile,))
    for k in range(20):
        w = sc.next_window(1000.0)
        assert np.all(np.diff(w.arrivals) >= 0)
        assert np.all((w.arrivals >= k * 1000) & (w.arrivals < (k + 1) * 1000))
        assert w.arrivals.shape == w.tx_ms.shape


def test_same_seed_same_trace():
    a = TrafficScenario(np.random.default_rng(42))
    b = TrafficScenario(np.random.default_rng(42))
    for _ in range(30):
        wa, wb = a.next_window(1000.0), b.next_window(1000.0)
        np.testing.assert_array_equal(wa.arrivals, wb.arrivals)
        assert wa.profile == wb.profile


def test_scenario_visits_multiple_profiles():
    sc = TrafficScenario(np.random.default_rng(0), mean_sojourn_windows=3)
    seen = {sc.next_window(1000.0).profile for _ in range(200)}
    assert seen == set(PROFILES)


def test_gaming_rate_is_roughly_right():
    sc = TrafficScenario(np.random.default_rng(0), ("gaming",))
    n = sum(sc.next_window(1000.0).arrivals.size for _ in range(100))
    assert 25 < n / 100 < 35


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        TrafficScenario(np.random.default_rng(0), ("netflix",))
