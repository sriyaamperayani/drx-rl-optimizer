import pytest

from drxgym.config import DRX_OFF, DRXConfig, PowerModel, RadioConfig, build_action_grid


def test_zero_gap_costs_nothing():
    assert PowerModel().sleep_energy(0) == 0.0


def test_short_gap_uses_micro_sleep():
    pm = PowerModel()
    assert pm.sleep_energy(3.0) == pytest.approx(pm.micro_sleep * 3.0)


def test_long_gap_prefers_deep_sleep():
    pm = PowerModel()
    gap = 300.0
    deep = pm.deep_sleep * (gap - pm.deep_sleep_transition_ms) + pm.deep_sleep_transition_energy
    assert pm.sleep_energy(gap) == pytest.approx(deep)


def test_sleep_is_always_cheaper_than_staying_awake():
    pm = PowerModel()
    for gap in range(1, 500):
        assert 0 < pm.sleep_energy(gap) < pm.pdcch_monitor * gap


def test_picks_cheapest_feasible_state():
    pm = PowerModel()
    gap = 10.0  # light sleep feasible, deep sleep not
    light = pm.light_sleep * (gap - pm.light_sleep_transition_ms) + pm.light_sleep_transition_energy
    assert pm.sleep_energy(gap) == pytest.approx(min(light, pm.micro_sleep * gap))


def test_cycle_must_fit_on_duration_and_beam_search():
    with pytest.raises(ValueError):
        DRXConfig(inactivity_ms=10, cycle_ms=10, on_duration_ms=8).validate(RadioConfig(beam_search_ms=4))


def test_action_grid_is_valid_and_unique():
    grid = build_action_grid()
    assert len(grid) == len(set(grid)) == 20
    for cfg in grid:
        cfg.validate(RadioConfig())


def test_drx_off_label():
    assert DRX_OFF.always_on and DRX_OFF.label == "drx_off"
