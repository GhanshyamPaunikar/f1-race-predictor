"""Pytest fixtures: synthetic F1 race data + trained predictor for API tests."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

# Make backend importable without installing
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def _synthetic_races(seed: int = 7) -> list[dict]:
    """Build a small but realistic-shaped dataset: 3 seasons × 5 rounds × 6 drivers."""
    rng = random.Random(seed)
    drivers = [
        ("hamilton", "mercedes"),
        ("verstappen", "red_bull"),
        ("leclerc", "ferrari"),
        ("norris", "mclaren"),
        ("alonso", "aston_martin"),
        ("sainz", "ferrari"),
    ]
    circuits = ["bahrain", "monza", "spa", "suzuka", "silverstone"]
    races = []
    for year in (2022, 2023, 2024):
        for rnd, cid in enumerate(circuits, start=1):
            grid = rng.sample(drivers, len(drivers))
            results = []
            # Final position is grid + noise — strong signal so the model is learnable
            for grid_idx, (did, tid) in enumerate(grid):
                final = max(1, min(20, grid_idx + 1 + rng.randint(-1, 1)))
                results.append({
                    "driver_id": did, "team_id": tid,
                    "position": final, "grid": grid_idx + 1,
                    "points": max(0, 25 - final * 3),
                    "status": "Finished" if rng.random() > 0.05 else "Engine",
                })
            races.append({
                "year": year, "round": rnd,
                "circuit_id": cid, "race_name": f"{cid.title()} GP",
                "date": f"{year}-0{rnd}-15", "total_rounds": len(circuits),
                "results": results,
            })
    return races


@pytest.fixture(scope="session")
def synthetic_races() -> list[dict]:
    return _synthetic_races()


@pytest.fixture(scope="session")
def trained_predictor(synthetic_races, tmp_path_factory):
    """Train once per test session — training takes a few seconds."""
    from predictor import MODEL_DIR, F1Predictor

    # Redirect model save to a tmp dir so we don't clobber the real one
    tmp_models = tmp_path_factory.mktemp("models")
    original = MODEL_DIR
    import predictor as p
    p.MODEL_DIR = tmp_models  # type: ignore[attr-defined]
    pred = F1Predictor()
    pred.train(synthetic_races)
    yield pred
    p.MODEL_DIR = original  # type: ignore[attr-defined]


@pytest.fixture
def sample_grid() -> list[dict]:
    return [
        {"driver_id": "verstappen", "team_id": "red_bull", "name": "Max Verstappen",
         "team": "Red Bull", "team_color": "#3671C6", "grid": 1},
        {"driver_id": "hamilton", "team_id": "mercedes", "name": "Lewis Hamilton",
         "team": "Mercedes", "team_color": "#00D2BE", "grid": 2},
        {"driver_id": "leclerc", "team_id": "ferrari", "name": "Charles Leclerc",
         "team": "Ferrari", "team_color": "#E8002D", "grid": 3},
        {"driver_id": "norris", "team_id": "mclaren", "name": "Lando Norris",
         "team": "McLaren", "team_color": "#FF8000", "grid": 4},
    ]
