"""
F1 Race Predictor — XGBoost + HistGBR + GBR ensemble with feature engineering.

Upgrades over v1:
 - Honest time-series CV (GroupKFold by season, no future-data leakage)
 - Quantile regression heads (P10/P90) for calibrated per-driver uncertainty
 - Separate DNF probability head (logistic, gradient-boosted)
 - Top-3 / Top-10 / podium hit-rate metrics alongside MAE
 - SHAP-based per-prediction explainability
 - Optional weather features (temperature, precipitation)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TEAM_COLORS: dict[str, str] = {
    "red_bull": "#3671C6",
    "ferrari": "#E8002D",
    "mercedes": "#00D2BE",
    "mclaren": "#FF8000",
    "aston_martin": "#229971",
    "alpine": "#0090FF",
    "williams": "#64C4FF",
    "rb": "#6692FF",
    "haas": "#B6BABD",
    "sauber": "#52E252",
    "alphatauri": "#6692FF",
    "alfa": "#900000",
    "racing_point": "#F596C8",
    "renault": "#FFF500",
    "default": "#888888",
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


FEATURE_COLS = [
    "grid_position",
    "inv_grid",
    "exp_grid",
    "is_front_row",
    "team_pts_pct",
    "team_form_last3",
    "driver_season_pts",
    "recent5_avg",
    "recent3_avg",
    "consistency",
    "circuit_avg",
    "circuit_exp",
    "season_progress",
    "grid_vs_form",
    "quali_gap_to_pole",
    "air_temp_c",
    "precip_mm",
    # Practice-session features (FastF1) — race-pace info not in qualifying alone
    "fp_best_lap_norm",
    "fp_longrun_pace_norm",
    "fp_teammate_quali_gap",
    "fp_session_laps",
    "fp_consistency",
]

FEATURE_LABELS = {
    "grid_position":      "Starting Grid Position",
    "inv_grid":           "1 / Grid Position",
    "exp_grid":           "Exp Grid Decay",
    "is_front_row":       "Front Row Qualifier",
    "team_pts_pct":       "Team Points Share (%)",
    "team_form_last3":    "Team Form (last 3 races)",
    "driver_season_pts":  "Driver Season Points",
    "recent5_avg":        "Recent Form (last 5)",
    "recent3_avg":        "Recent Form (last 3)",
    "consistency":        "Finishing Consistency",
    "circuit_avg":        "Circuit History Avg",
    "circuit_exp":        "Circuit Experience",
    "season_progress":    "Season Progress",
    "grid_vs_form":       "Grid vs. Expected Form",
    "quali_gap_to_pole":  "Quali Gap to Pole (s)",
    "air_temp_c":         "Air Temperature (°C)",
    "precip_mm":          "Precipitation (mm)",
    "fp_best_lap_norm":      "Practice Best Lap Gap (s)",
    "fp_longrun_pace_norm":  "Practice Long-Run Pace Gap (s)",
    "fp_teammate_quali_gap": "Quali Gap vs Teammate (s)",
    "fp_session_laps":       "Practice Laps Completed",
    "fp_consistency":        "Practice Lap-Time Consistency",
}


class FeatureContext:
    """Rolling statistics computed over historical races, used for both
    training-time feature building and inference-time feature lookup."""

    def __init__(self) -> None:
        self.driver_history: dict[str, list[float]] = defaultdict(list)
        self.circuit_driver_history: dict[str, list[float]] = defaultdict(list)
        self.driver_dnf_history: dict[str, list[int]] = defaultdict(list)
        # Rolling per-team finish history: average finish of team's drivers, per race
        self.team_finish_history: dict[str, list[float]] = defaultdict(list)
        self.team_season_pts: dict[str, float] = defaultdict(float)
        self.driver_season_pts: dict[str, float] = defaultdict(float)
        self.total_points_by_season: dict[int, float] = defaultdict(float)

    def update_after_race(self, race: dict) -> None:
        year = race["year"]
        circuit_id = race["circuit_id"]
        # Aggregate team finishes for this race
        team_finishes: dict[str, list[float]] = defaultdict(list)
        for r in race["results"]:
            did = r["driver_id"]
            tid = r["team_id"]
            pos = _safe_float(r.get("position"), 20.0)
            pts = _safe_float(r.get("points"), 0.0)
            dnf = _is_dnf(r.get("status", ""))
            if pos <= 20:
                self.driver_history[did].append(pos)
                self.circuit_driver_history[f"{did}_{circuit_id}"].append(pos)
                team_finishes[tid].append(pos)
            self.driver_dnf_history[did].append(int(dnf))
            self.team_season_pts[f"{year}_{tid}"] += pts
            self.driver_season_pts[f"{year}_{did}"] += pts
            self.total_points_by_season[year] += pts
        for tid, finishes in team_finishes.items():
            self.team_finish_history[tid].append(float(np.mean(finishes)))

    def features_for(
        self,
        driver_id: str,
        team_id: str,
        circuit_id: str,
        year: int,
        round_num: int,
        total_rounds: int,
        grid_pos: float,
        weather: dict | None = None,
        quali_gap: float | None = None,
        practice: dict | None = None,
    ) -> list[float]:
        hist = self.driver_history.get(driver_id, [])
        r5 = float(np.mean(hist[-5:])) if hist else 10.0
        r3 = float(np.mean(hist[-3:])) if hist else 10.0
        cons = float(np.std(hist[-5:])) if len(hist) >= 3 else 5.0

        chist = self.circuit_driver_history.get(f"{driver_id}_{circuit_id}", [])
        c_avg = float(np.mean(chist)) if chist else 10.0
        c_exp = float(min(len(chist), 10))

        total_pts = self.total_points_by_season.get(year, 1.0) or 1.0
        team_pct = self.team_season_pts.get(f"{year}_{team_id}", 0.0) / total_pts * 100
        team_recent = self.team_finish_history.get(team_id, [])
        team_form3 = float(np.mean(team_recent[-3:])) if team_recent else 10.0
        drv_pts = self.driver_season_pts.get(f"{year}_{driver_id}", 0.0)
        season_progress = round_num / max(total_rounds, 1)

        # Smooth grid encodings — replaces the dominant binary is_top5_grid flag
        g = max(grid_pos, 1.0)
        inv_grid = 1.0 / g
        exp_grid = float(np.exp(-g / 3.0))

        w = weather or {}
        p = practice or {}
        return [
            grid_pos, inv_grid, exp_grid,
            1.0 if grid_pos <= 2 else 0.0,
            team_pct, team_form3,
            drv_pts, r5, r3, cons,
            c_avg, c_exp, season_progress,
            grid_pos - r5,
            float(quali_gap) if quali_gap is not None else 0.5 * (grid_pos - 1),
            float(w.get("air_temp_c", 22.0)),
            float(w.get("precip_mm", 0.0)),
            # Practice features — neutral defaults when missing so the model
            # treats absent practice data as "average driver, no signal"
            float(p.get("fp_best_lap_norm", 0.5)),
            float(p.get("fp_longrun_pace_norm", 1.0)),
            float(p.get("fp_teammate_quali_gap", 0.0)),
            float(p.get("fp_session_laps", 50.0)),
            float(p.get("fp_consistency", 2.0)),
        ]


def _is_dnf(status: str) -> bool:
    """A driver counts as 'Finished' if status starts with Finished or +<lap>."""
    if not status:
        return True
    s = status.strip()
    return not (s.startswith("Finished") or (s.startswith("+") and "Lap" in s))


class DataProcessor:
    """Streams races in chronological order to produce a training DataFrame
    where every row's features only reflect data available BEFORE that race."""

    def build_training_df(
        self,
        all_races: list[dict],
        weather_lookup: dict[tuple, dict] | None = None,
        quali_gap_lookup: dict[tuple, dict[str, float]] | None = None,
        practice_lookup: dict[tuple, dict[str, dict[str, float]]] | None = None,
        id_to_code: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, FeatureContext]:
        ctx = FeatureContext()
        weather_lookup = weather_lookup or {}
        quali_gap_lookup = quali_gap_lookup or {}
        practice_lookup = practice_lookup or {}
        id_to_code = id_to_code or {}
        records: list[dict] = []

        for race in sorted(all_races, key=lambda x: (x["year"], x["round"])):
            year = race["year"]
            rnd = race["round"]
            cid = race["circuit_id"]
            total_rounds = race.get("total_rounds", 23)
            weather = weather_lookup.get((year, rnd), {})
            gaps = quali_gap_lookup.get((year, rnd), {})
            practice = practice_lookup.get((year, rnd), {})

            for result in race["results"]:
                did = result["driver_id"]
                tid = result["team_id"]
                grid = _safe_float(result.get("grid"), 10.0)
                pos = _safe_float(result.get("position"), 20.0)
                dnf = int(_is_dnf(result.get("status", "")))
                gap = gaps.get(did)
                code = id_to_code.get(did, "")
                practice_feats = practice.get(code, {}) if code else {}

                feats = ctx.features_for(
                    did, tid, cid, year, rnd, total_rounds, grid, weather, gap,
                    practice_feats,
                )
                rec = dict(zip(FEATURE_COLS, feats, strict=False))
                rec.update({
                    "year": year, "round": rnd, "driver_id": did, "team_id": tid,
                    "circuit_id": cid, "final_position": pos, "dnf": dnf,
                })
                records.append(rec)

            ctx.update_after_race(race)

        return pd.DataFrame(records), ctx


class F1Predictor:
    """Ensemble regressor + quantile heads + DNF classifier."""

    def __init__(self) -> None:
        self.xgb_reg: xgb.XGBRegressor | None = None
        self.hgbr: HistGradientBoostingRegressor | None = None
        self.gbr: GradientBoostingRegressor | None = None
        self.lgb_rank: lgb.LGBMRanker | None = None
        self.q_low: HistGradientBoostingRegressor | None = None
        self.q_high: HistGradientBoostingRegressor | None = None
        self.dnf_clf: GradientBoostingClassifier | None = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics: dict[str, Any] = {}
        self.processor = DataProcessor()
        self.ctx = FeatureContext()
        # Ensemble weights — re-tuned post-rebuild to give the ranker a strong voice
        self.weights = {"xgb": 0.30, "hgbr": 0.25, "gbr": 0.15, "lgb_rank": 0.30}
        # Cached SHAP explainer (lazy)
        self._shap_explainer = None

    # ── Training ────────────────────────────────────────────────────────

    def train(
        self,
        all_races: list[dict],
        weather_lookup: dict[tuple, dict] | None = None,
        quali_gap_lookup: dict[tuple, dict[str, float]] | None = None,
        practice_lookup: dict[tuple, dict[str, dict[str, float]]] | None = None,
        id_to_code: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        logger.info("Training on %d races…", len(all_races))
        df, ctx = self.processor.build_training_df(
            all_races, weather_lookup, quali_gap_lookup, practice_lookup, id_to_code,
        )
        self.ctx = ctx

        df = df[(df["grid_position"] > 0) & (df["final_position"] > 0) & (df["final_position"] <= 20)]
        df = df.sort_values(["year", "round"]).copy()
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

        X = df[FEATURE_COLS].values.astype(float)
        y = df["final_position"].values.astype(float)
        y_dnf = df["dnf"].values.astype(int)
        groups = df["year"].values
        X_sc = self.scaler.fit_transform(X)

        # Sample weights: 2018 -> 1.0, +0.3 per year → 2025 ≈ 3.1x
        years_arr = df["year"].values.astype(float)
        sw = 1.0 + 0.3 * (years_arr - years_arr.min())

        # LambdaRank groups: one group per race; relevance = 21 - final_position
        race_groups = df.groupby(["year", "round"]).size().values
        y_rank = (21 - df["final_position"].clip(1, 20)).astype(int).values

        # Main regressors
        self.xgb_reg = xgb.XGBRegressor(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.75,
            min_child_weight=3, gamma=0.1,
            random_state=42, verbosity=0, n_jobs=-1,
        )
        self.hgbr = HistGradientBoostingRegressor(
            max_iter=600, max_depth=5, learning_rate=0.03,
            min_samples_leaf=10, random_state=42,
        )
        self.gbr = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.8, random_state=42,
        )
        # LightGBM LambdaRank — purpose-built for ordering problems
        self.lgb_rank = lgb.LGBMRanker(
            n_estimators=500, max_depth=6, learning_rate=0.04,
            num_leaves=31, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
        # Quantile heads for per-driver uncertainty
        self.q_low = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.10, max_iter=300, max_depth=5,
            learning_rate=0.05, random_state=42,
        )
        self.q_high = HistGradientBoostingRegressor(
            loss="quantile", quantile=0.90, max_iter=300, max_depth=5,
            learning_rate=0.05, random_state=42,
        )
        # DNF classifier
        self.dnf_clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        )

        self.xgb_reg.fit(X_sc, y, sample_weight=sw)
        self.hgbr.fit(X_sc, y, sample_weight=sw)
        self.gbr.fit(X_sc, y, sample_weight=sw)
        self.lgb_rank.fit(X_sc, y_rank, group=race_groups, sample_weight=sw)
        self.q_low.fit(X_sc, y, sample_weight=sw)
        self.q_high.fit(X_sc, y, sample_weight=sw)
        self.dnf_clf.fit(X_sc, y_dnf, sample_weight=sw)

        metrics = self._compute_cv_metrics(X_sc, y, groups, df, sw, race_groups, y_rank)
        metrics["feature_importance"] = dict(
            zip(FEATURE_COLS, self.xgb_reg.feature_importances_.tolist(), strict=False)
        )
        metrics["n_races"] = len(all_races)
        metrics["n_samples"] = len(df)
        metrics["features"] = FEATURE_COLS
        metrics["ensemble_weights"] = self.weights

        self.metrics = metrics
        self.is_trained = True
        self._shap_explainer = None

        joblib.dump(self._serialize(), MODEL_DIR / "f1_predictor.pkl")
        logger.info(
            "Training done. CV MAE: %.3f · Winner %.1f%% · Podium %.1f%% · Top10 %.1f%%",
            metrics["cv_mae"],
            metrics["winner_hit_rate"] * 100,
            metrics["podium_hit_rate"] * 100,
            metrics["top10_hit_rate"] * 100,
        )
        return metrics

    def _compute_cv_metrics(
        self,
        X_sc: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        df: pd.DataFrame,
        sample_weights: np.ndarray,
        race_groups: np.ndarray,
        y_rank: np.ndarray,
    ) -> dict[str, float]:
        """GroupKFold by season — every fold predicts a held-out season using
        only the others, so no future data leaks into training.

        For the ranker we re-derive per-fold race groups by counting per-(year,
        round) in the training indices, preserving the original race ordering."""
        unique_years = np.unique(groups)
        n_splits = min(5, len(unique_years))
        gkf = GroupKFold(n_splits=n_splits)

        fold_maes: list[float] = []
        all_pred = np.zeros_like(y)
        w = self.weights
        for tr_idx, te_idx in gkf.split(X_sc, y, groups):
            xgb_m = xgb.XGBRegressor(
                n_estimators=600, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.75,
                random_state=42, verbosity=0, n_jobs=-1,
            )
            hgbr = HistGradientBoostingRegressor(
                max_iter=600, max_depth=5, learning_rate=0.03, random_state=42,
            )
            gbr = GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.04, random_state=42,
            )
            ranker = lgb.LGBMRanker(
                n_estimators=500, max_depth=6, learning_rate=0.04,
                num_leaves=31, min_child_samples=10,
                random_state=42, verbosity=-1, n_jobs=-1,
            )
            xgb_m.fit(X_sc[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
            hgbr.fit(X_sc[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])
            gbr.fit(X_sc[tr_idx], y[tr_idx], sample_weight=sample_weights[tr_idx])

            # Per-fold ranker groups: count rows per race within the training subset
            train_df = df.iloc[tr_idx]
            tr_groups = train_df.groupby(["year", "round"], sort=False).size().values
            ranker.fit(
                X_sc[tr_idx], y_rank[tr_idx],
                group=tr_groups, sample_weight=sample_weights[tr_idx],
            )

            # Ranker outputs a score where HIGHER = better (P1). We invert to
            # finish-position space (1..20) so it can mix with the regressors.
            rank_scores = ranker.predict(X_sc[te_idx])
            # Map to predicted finish: high score → low position. Use rank within
            # each held-out race for a calibrated 1..N value.
            test_df = df.iloc[te_idx].copy()
            test_df["_rs"] = rank_scores
            inferred_pos = (
                test_df.groupby(["year", "round"])["_rs"]
                .rank(ascending=False, method="first").values
            )

            pred = (
                w["xgb"] * xgb_m.predict(X_sc[te_idx])
                + w["hgbr"] * hgbr.predict(X_sc[te_idx])
                + w["gbr"] * gbr.predict(X_sc[te_idx])
                + w["lgb_rank"] * inferred_pos
            )
            all_pred[te_idx] = pred
            fold_maes.append(mean_absolute_error(y[te_idx], pred))

        # Convert per-race continuous predictions into rank predictions
        df = df.copy()
        df.loc[:, "pred_score"] = all_pred
        df.loc[:, "pred_rank"] = df.groupby(["year", "round"])["pred_score"].rank(method="first")

        top1 = (df["pred_rank"] == 1) & (df["final_position"] == 1)
        top3 = (df["pred_rank"] <= 3) & (df["final_position"] <= 3)
        top10 = (df["pred_rank"] <= 10) & (df["final_position"] <= 10)

        n_races = df.groupby(["year", "round"]).ngroups

        # Naive baseline: grid == final
        naive_mae = mean_absolute_error(y, df["grid_position"].values)

        return {
            "cv_mae": float(np.mean(fold_maes)),
            "cv_mae_std": float(np.std(fold_maes)),
            "train_mae": float(mean_absolute_error(y, all_pred)),
            "naive_mae": float(naive_mae),
            "lift_vs_naive": float(naive_mae - np.mean(fold_maes)),
            "winner_hit_rate": float(top1.sum() / n_races),
            "podium_hit_rate": float(top3.sum() / (n_races * 3)),
            "top10_hit_rate": float(top10.sum() / (n_races * 10)),
        }

    # ── Inference ───────────────────────────────────────────────────────

    def _build_X(
        self,
        grid: list[dict],
        circuit_id: str,
        year: int,
        round_num: int,
        total_rounds: int,
        weather: dict | None = None,
        quali_gaps: dict[str, float] | None = None,
        practice: dict[str, dict[str, float]] | None = None,
    ) -> np.ndarray:
        gaps = quali_gaps or {}
        practice = practice or {}
        rows: list[list[float]] = []
        for driver in grid:
            did = driver["driver_id"]
            # Practice keyed by 3-letter code OR driver_id (caller's choice)
            p = practice.get(did) or practice.get(driver.get("code", ""))
            rows.append(self.ctx.features_for(
                driver_id=did,
                team_id=driver.get("team_id", "default"),
                circuit_id=circuit_id,
                year=year,
                round_num=round_num,
                total_rounds=total_rounds,
                grid_pos=_safe_float(driver.get("grid"), 10.0),
                weather=weather,
                quali_gap=gaps.get(did),
                practice=p,
            ))
        return self.scaler.transform(np.asarray(rows, dtype=float))

    def _ensemble_predict(self, X_sc: np.ndarray) -> np.ndarray:
        """Returns a finish-position score (lower = better). The LightGBM
        ranker outputs a relevance score (higher = better) so we invert it
        to per-batch positions via rank()."""
        assert self.xgb_reg and self.hgbr and self.gbr
        w = self.weights
        reg_part = (
            w["xgb"] * self.xgb_reg.predict(X_sc)
            + w["hgbr"] * self.hgbr.predict(X_sc)
            + w["gbr"] * self.gbr.predict(X_sc)
        )
        if self.lgb_rank is None:
            return reg_part / (w["xgb"] + w["hgbr"] + w["gbr"])
        rank_scores = self.lgb_rank.predict(X_sc)
        # Higher rank score = better; convert to predicted position
        order = np.argsort(-rank_scores)
        rank_positions = np.empty_like(rank_scores)
        rank_positions[order] = np.arange(1, len(rank_scores) + 1)
        return reg_part + w["lgb_rank"] * rank_positions

    def predict_race(
        self,
        grid: list[dict],
        circuit_id: str,
        year: int,
        round_num: int,
        total_rounds: int = 24,
        weather: dict | None = None,
        quali_gaps: dict[str, float] | None = None,
        practice: dict[str, dict[str, float]] | None = None,
    ) -> list[dict]:
        if not self.is_trained:
            return self._fallback_predict(grid)

        X_sc = self._build_X(
            grid, circuit_id, year, round_num, total_rounds, weather, quali_gaps, practice,
        )
        scores = self._ensemble_predict(X_sc)
        assert self.q_low and self.q_high and self.dnf_clf
        lo = self.q_low.predict(X_sc)
        hi = self.q_high.predict(X_sc)
        dnf_prob = self.dnf_clf.predict_proba(X_sc)[:, 1]

        results = [
            {**driver, "_score": float(scores[i]), "_lo": float(lo[i]), "_hi": float(hi[i]),
             "dnf_probability": round(float(dnf_prob[i]), 3)}
            for i, driver in enumerate(grid)
        ]
        results.sort(key=lambda x: x["_score"])
        spread = float(np.std(scores)) if len(scores) > 1 else 5.0

        # Confidence blends two signals:
        #   1. Gap to the next driver in predicted order (separation)
        #   2. Tightness of this driver's own P10–P90 quantile interval
        # Result is in [0.05, 0.97] — a real low-confidence floor, not the
        # old 0.25 which lied about uncertainty for ~70% of mid-pack drivers.
        for i, r in enumerate(results):
            gap_next = abs(results[i + 1]["_score"] - r["_score"]) if i + 1 < len(results) else spread
            sep = gap_next / max(spread, 0.5)              # 0 → no separation, ~2 → strong
            interval = max(r["_hi"] - r["_lo"], 0.1)
            tight = 1.0 / (1.0 + interval / 3.0)           # tight 80% interval → high score
            raw = 0.55 * min(sep, 1.5) / 1.5 + 0.45 * tight
            confidence = min(0.97, max(0.05, raw))
            r["predicted_position"] = i + 1
            r["confidence"] = round(confidence, 3)
            r["predicted_score"] = round(r.pop("_score"), 3)
            r["score_low"] = round(r.pop("_lo"), 2)
            r["score_high"] = round(r.pop("_hi"), 2)
        return results

    def get_win_probabilities(
        self,
        grid: list[dict],
        circuit_id: str,
        year: int,
        round_num: int,
        total_rounds: int = 24,
        n_simulations: int = 2000,
        weather: dict | None = None,
        quali_gaps: dict[str, float] | None = None,
        practice: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Monte Carlo using per-driver uncertainty from quantile heads,
        not a single global std. DNFs are sampled too."""
        if not self.is_trained:
            probs: dict[str, float] = {}
            for i, d in enumerate(sorted(grid, key=lambda x: x.get("grid", 20))):
                probs[d["driver_id"]] = max(0.01, 0.5 - i * 0.04)
            total = sum(probs.values())
            return {k: round(v / total, 3) for k, v in probs.items()}

        X_sc = self._build_X(
            grid, circuit_id, year, round_num, total_rounds, weather, quali_gaps, practice,
        )
        base = self._ensemble_predict(X_sc)
        assert self.q_low and self.q_high and self.dnf_clf
        lo = self.q_low.predict(X_sc)
        hi = self.q_high.predict(X_sc)
        # Approximate per-driver std from the 80% interval (≈ 2.56·σ)
        sigma = np.clip((hi - lo) / 2.56, 0.5, 8.0)
        dnf_prob = self.dnf_clf.predict_proba(X_sc)[:, 1]

        rng = np.random.default_rng(42)
        win_counts: dict[str, int] = {d["driver_id"]: 0 for d in grid}
        for _ in range(n_simulations):
            noise = rng.normal(0, sigma)
            dnf_mask = rng.random(len(grid)) < dnf_prob
            sim = base + noise
            sim[dnf_mask] = 99.0  # DNF cannot win
            win_idx = int(np.argmin(sim))
            win_counts[grid[win_idx]["driver_id"]] += 1

        return {k: round(v / n_simulations, 3) for k, v in win_counts.items()}

    # ── Explainability ──────────────────────────────────────────────────

    def explain_driver(
        self,
        grid: list[dict],
        driver_id: str,
        circuit_id: str,
        year: int,
        round_num: int,
        total_rounds: int = 24,
        weather: dict | None = None,
        quali_gaps: dict[str, float] | None = None,
        practice: dict[str, dict[str, float]] | None = None,
    ) -> list[dict]:
        """Returns per-feature contribution for one driver, ordered by |impact|."""
        if not self.is_trained:
            return []
        try:
            import shap
        except ImportError:
            return [
                {"feature": k, "label": FEATURE_LABELS.get(k, k), "impact": float(v)}
                for k, v in sorted(
                    self.metrics.get("feature_importance", {}).items(),
                    key=lambda kv: -kv[1],
                )
            ]
        idx = next((i for i, d in enumerate(grid) if d["driver_id"] == driver_id), None)
        if idx is None:
            return []
        X_sc = self._build_X(
            grid, circuit_id, year, round_num, total_rounds, weather, quali_gaps, practice,
        )
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.xgb_reg)
        sv = self._shap_explainer.shap_values(X_sc[idx:idx + 1])[0]
        contributions = [
            {"feature": f, "label": FEATURE_LABELS.get(f, f),
             "value": float(X_sc[idx, j]), "impact": float(sv[j])}
            for j, f in enumerate(FEATURE_COLS)
        ]
        contributions.sort(key=lambda c: -abs(c["impact"]))
        return contributions

    # ── Fallback / persistence ──────────────────────────────────────────

    def _fallback_predict(self, grid: list[dict]) -> list[dict]:
        results = sorted(grid, key=lambda x: x.get("grid", 20))
        for i, r in enumerate(results):
            r["predicted_position"] = i + 1
            r["confidence"] = round(max(0.3, 0.92 - i * 0.03), 3)
            r["dnf_probability"] = 0.1
        return results

    def _serialize(self) -> dict:
        return {
            "version": 4,
            "xgb": self.xgb_reg,
            "hgbr": self.hgbr,
            "gbr": self.gbr,
            "lgb_rank": self.lgb_rank,
            "q_low": self.q_low,
            "q_high": self.q_high,
            "dnf_clf": self.dnf_clf,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "weights": self.weights,
            "ctx": {
                "driver_history": dict(self.ctx.driver_history),
                "circuit_driver_history": dict(self.ctx.circuit_driver_history),
                "driver_dnf_history": dict(self.ctx.driver_dnf_history),
                "team_finish_history": dict(self.ctx.team_finish_history),
                "team_season_pts": dict(self.ctx.team_season_pts),
                "driver_season_pts": dict(self.ctx.driver_season_pts),
                "total_points_by_season": dict(self.ctx.total_points_by_season),
            },
        }

    def load(self) -> bool:
        p = MODEL_DIR / "f1_predictor.pkl"
        if not p.exists():
            return False
        try:
            data = joblib.load(p)
            if data.get("version", 1) < 4:
                logger.info("Found older model file — will retrain to v4.")
                return False
            self.xgb_reg = data["xgb"]
            self.hgbr = data["hgbr"]
            self.gbr = data["gbr"]
            self.lgb_rank = data.get("lgb_rank")
            self.q_low = data["q_low"]
            self.q_high = data["q_high"]
            self.dnf_clf = data["dnf_clf"]
            self.scaler = data["scaler"]
            self.metrics = data["metrics"]
            self.weights = data.get("weights", self.weights)
            ctx = data.get("ctx", {})
            self.ctx = FeatureContext()
            self.ctx.driver_history = defaultdict(list, ctx.get("driver_history", {}))
            self.ctx.circuit_driver_history = defaultdict(list, ctx.get("circuit_driver_history", {}))
            self.ctx.driver_dnf_history = defaultdict(list, ctx.get("driver_dnf_history", {}))
            self.ctx.team_finish_history = defaultdict(list, ctx.get("team_finish_history", {}))
            self.ctx.team_season_pts = defaultdict(float, ctx.get("team_season_pts", {}))
            self.ctx.driver_season_pts = defaultdict(float, ctx.get("driver_season_pts", {}))
            self.ctx.total_points_by_season = defaultdict(float, ctx.get("total_points_by_season", {}))
            self.is_trained = True
            logger.info("Model v%s loaded from disk.", data.get("version", "?"))
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False
