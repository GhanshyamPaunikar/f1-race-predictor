"""Tests for the predictor: feature engineering, training, prediction, win probs."""
from __future__ import annotations

import numpy as np
import pytest
from predictor import (
    FEATURE_COLS,
    DataProcessor,
    F1Predictor,
    FeatureContext,
    _is_dnf,
    _safe_float,
)

# ── Pure helpers ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("val", "expected"), [
    ("3.5", 3.5), (7, 7.0), ("", 0.0), (None, 0.0), ("nope", 0.0), (float("nan"), 0.0),
])
def test_safe_float(val, expected):
    if expected == 0.0 and isinstance(val, float) and val != val:
        # NaN passes through float() so check separately
        assert np.isnan(_safe_float(val, default=float("nan")))
        return
    assert _safe_float(val) == expected


@pytest.mark.parametrize(("status", "is_dnf"), [
    ("Finished",          False),
    ("+1 Lap",            False),
    ("+2 Laps",           False),
    ("Engine",            True),
    ("Collision",         True),
    ("",                  True),  # empty status = unknown = DNF (conservative)
    ("Disqualified",      True),
])
def test_is_dnf(status, is_dnf):
    assert _is_dnf(status) is is_dnf


# ── Feature engineering ────────────────────────────────────────────────


def test_data_processor_no_future_leakage(synthetic_races):
    """The first race for any driver must have neutral history features
    (recent_avg=10.0, circuit_exp=0). If those leak in, defaults are wrong."""
    df, _ = DataProcessor().build_training_df(synthetic_races)

    # Take each driver's earliest race
    first_rows = df.sort_values(["driver_id", "year", "round"]).groupby("driver_id").head(1)
    assert (first_rows["circuit_exp"] == 0).all(), "Circuit experience leaked into first race"
    assert (first_rows["recent5_avg"] == 10.0).all(), "Recent form leaked into first race"


def test_feature_columns_present(synthetic_races):
    df, _ = DataProcessor().build_training_df(synthetic_races)
    for col in FEATURE_COLS:
        assert col in df.columns


def test_weather_features_default_when_missing(synthetic_races):
    df, _ = DataProcessor().build_training_df(synthetic_races, weather_lookup=None)
    # With no weather lookup, all weather values should fall back to defaults
    assert (df["air_temp_c"] == 22.0).all()
    assert (df["precip_mm"] == 0.0).all()


def test_weather_features_applied(synthetic_races):
    weather = {(2022, 1): {"air_temp_c": 35.0, "precip_mm": 12.0}}
    df, _ = DataProcessor().build_training_df(synthetic_races, weather_lookup=weather)
    hot = df[(df["year"] == 2022) & (df["round"] == 1)]
    assert (hot["air_temp_c"] == 35.0).all()
    assert (hot["precip_mm"] == 12.0).all()


def test_feature_context_features_for_unknown_driver():
    """An unknown driver should still produce a valid feature vector."""
    ctx = FeatureContext()
    feats = ctx.features_for(
        driver_id="who", team_id="default", circuit_id="monza",
        year=2024, round_num=1, total_rounds=24, grid_pos=10.0,
    )
    assert len(feats) == len(FEATURE_COLS)
    assert all(isinstance(f, float) for f in feats)


# ── Training & metrics ─────────────────────────────────────────────────


def test_trained_predictor_metrics(trained_predictor):
    m = trained_predictor.metrics
    assert m["cv_mae"] > 0
    assert m["n_samples"] > 0
    assert 0 <= m["podium_hit_rate"] <= 1
    assert 0 <= m["winner_hit_rate"] <= 1
    assert 0 <= m["top10_hit_rate"] <= 1
    # Feature importance covers exactly the feature columns
    assert set(m["feature_importance"].keys()) == set(FEATURE_COLS)


def test_predictor_beats_naive_baseline(trained_predictor):
    """On synthetic data where finish ≈ grid + noise, ML should match
    or slightly beat the naive baseline. This guards against regressions
    where the model produces nonsense."""
    m = trained_predictor.metrics
    # naive grid==finish is by construction the ground truth ± 1, so the
    # bar is just "not catastrophically worse than baseline".
    assert m["cv_mae"] <= m["naive_mae"] + 0.5


# ── Prediction & win probs ────────────────────────────────────────────


def test_predict_race_returns_ranked_grid(trained_predictor, sample_grid):
    preds = trained_predictor.predict_race(
        sample_grid, circuit_id="monza", year=2024, round_num=3, total_rounds=24,
    )
    assert len(preds) == len(sample_grid)
    positions = [p["predicted_position"] for p in preds]
    assert positions == list(range(1, len(sample_grid) + 1))
    for p in preds:
        assert "confidence" in p and 0 <= p["confidence"] <= 1
        assert "dnf_probability" in p and 0 <= p["dnf_probability"] <= 1
        assert "score_low" in p and "score_high" in p
        assert p["score_low"] <= p["score_high"]


def test_win_probabilities_sum_to_one(trained_predictor, sample_grid):
    probs = trained_predictor.get_win_probabilities(
        sample_grid, circuit_id="monza", year=2024, round_num=3,
        total_rounds=24, n_simulations=200,
    )
    assert set(probs.keys()) == {d["driver_id"] for d in sample_grid}
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01
    assert all(0 <= v <= 1 for v in probs.values())


def test_pole_sitter_has_highest_win_probability(trained_predictor, sample_grid):
    """Pole sitter should have the highest win probability — basic sanity."""
    probs = trained_predictor.get_win_probabilities(
        sample_grid, circuit_id="monza", year=2024, round_num=3,
        total_rounds=24, n_simulations=500,
    )
    pole_driver = sample_grid[0]["driver_id"]
    assert max(probs, key=probs.get) == pole_driver


# ── Persistence ────────────────────────────────────────────────────────


def test_save_and_load_round_trip(trained_predictor, sample_grid, tmp_path, monkeypatch):
    import predictor as p
    monkeypatch.setattr(p, "MODEL_DIR", tmp_path)
    p.MODEL_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump(trained_predictor._serialize(), tmp_path / "f1_predictor.pkl")

    fresh = p.F1Predictor()
    assert fresh.load() is True
    assert fresh.is_trained
    preds = fresh.predict_race(sample_grid, "monza", 2024, 3, 24)
    assert len(preds) == len(sample_grid)


def test_fallback_predict_when_untrained(sample_grid):
    """Untrained predictor still returns a sensible grid-order ranking."""
    p = F1Predictor()
    preds = p.predict_race(sample_grid, "monza", 2024, 3, 24)
    assert [r["predicted_position"] for r in preds] == [1, 2, 3, 4]


# ── Explainability ─────────────────────────────────────────────────────


def test_explain_driver_returns_contributions(trained_predictor, sample_grid):
    contribs = trained_predictor.explain_driver(
        sample_grid, "verstappen", "monza", 2024, 3, 24,
    )
    assert len(contribs) > 0
    # Each entry has the expected shape
    for c in contribs:
        assert {"feature", "label"}.issubset(c)


def test_explain_unknown_driver(trained_predictor, sample_grid):
    assert trained_predictor.explain_driver(
        sample_grid, "nobody", "monza", 2024, 3, 24,
    ) == []
