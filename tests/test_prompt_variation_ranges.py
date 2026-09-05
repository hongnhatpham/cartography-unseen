"""Independent sampling stays inside the visitor's inclusive variation ranges."""
import random

from app import main


def test_variation_includes_both_endpoints(monkeypatch):
    monkeypatch.setattr(main.secrets, "randbelow", lambda count: 0)
    assert main.random_prompt_settings() == {
        "timestep_min": 80, "timestep_max": 280, "display_sharpen": 1.,
        "instability": 0., "guide_strength": .65,
    }
    monkeypatch.setattr(main.secrets, "randbelow", lambda count: count - 1)
    assert main.random_prompt_settings() == {
        "timestep_min": 200, "timestep_max": 400, "display_sharpen": 2.,
        "instability": .4, "guide_strength": 1.,
    }


def test_combinations_vary_independently(monkeypatch):
    rng = random.Random(42)
    monkeypatch.setattr(main.secrets, "randbelow", rng.randrange)
    draws = [main.random_prompt_settings() for _ in range(200)]
    for values in draws:
        assert 80 <= values["timestep_min"] <= 200
        assert 280 <= values["timestep_max"] <= 400
        assert 1. <= values["display_sharpen"] <= 2.
        assert 0. <= values["instability"] <= .4
        assert .65 <= values["guide_strength"] <= 1.
    assert len({tuple(values.values()) for values in draws}) == len(draws)
    # High and low guide strengths can pair with either end of instability.
    assert {(values["guide_strength"] >= .825, values["instability"] >= .2)
            for values in draws} == {(False, False), (False, True), (True, False), (True, True)}
    assert {(values["timestep_min"] >= 140, values["timestep_max"] >= 340)
            for values in draws} == {(False, False), (False, True), (True, False), (True, True)}
