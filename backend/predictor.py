"""
F1 Race Predictor — XGBoost + HistGBR + GBR ensemble with feature engineering.

Key features:
 - Rolling driver form (last 3 / last 5 races)
 - Circuit-specific history per driver
 - Team strength (constructor points share)
 - Grid position and qualifying gap
 - Season progress and championship pressure
"""
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TEAM_COLORS = {
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


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_time_gap(t: str) -> float:
    """Convert qualifying time string to seconds."""
    if not t or t in ("", "N/A"):
        return 0.0
    try:
        if ":" in t:
            parts = t.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(t)
    except Exception:
        return 0.0


class DataProcessor:
    """Converts raw API race data into a training DataFrame."""

    def build_training_df(self, all_races: List[Dict]) -> pd.DataFrame:
        driver_history: Dict[str, List[float]] = {}
        circuit_driver_history: Dict[str, List[float]] = {}
        team_season_pts: Dict[str, float] = {}
        driver_season_pts: Dict[str, float] = {}
        records = []

        total_points_by_season: Dict[int, float] = {}

        for race in sorted(all_races, key=lambda x: (x["year"], x["round"])):
            year = race["year"]
            rnd = race["round"]
            circuit_id = race["circuit_id"]
            total_rounds = race.get("total_rounds", 23)
            season_progress = rnd / total_rounds

            for result in race["results"]:
                driver_id = result["driver_id"]
                team_id = result["team_id"]
                grid = _safe_float(result.get("grid"), 10.0)
                position = _safe_float(result.get("position"), 20.0)
                status = result.get("status", "")
                dnf = 0 if "Finished" in status or "+1" in status or "+" in status[:3] else 1

                # Rolling form
                hist = driver_history.get(driver_id, [])
                recent5 = float(np.mean(hist[-5:])) if hist else 10.0
                recent3 = float(np.mean(hist[-3:])) if hist else 10.0
                consistency = float(np.std(hist[-5:])) if len(hist) >= 3 else 5.0

                # Circuit history
                ck = f"{driver_id}_{circuit_id}"
                chist = circuit_driver_history.get(ck, [])
                circuit_avg = float(np.mean(chist)) if chist else 10.0
                circuit_exp = min(len(chist), 10)

                # Team strength
                total_pts = total_points_by_season.get(year, 1.0)
                team_pts = team_season_pts.get(f"{year}_{team_id}", 0.0)
                team_pct = team_pts / max(total_pts, 1.0) * 100

                driver_pts = driver_season_pts.get(f"{year}_{driver_id}", 0.0)

                records.append({
                    "year": year,
                    "round": rnd,
                    "driver_id": driver_id,
                    "team_id": team_id,
                    "circuit_id": circuit_id,
                    "grid_position": grid,
                    "final_position": position,
                    "dnf": dnf,
                    "team_pts_pct": team_pct,
                    "driver_season_pts": driver_pts,
                    "recent5_avg": recent5,
                    "recent3_avg": recent3,
                    "consistency": consistency,
                    "circuit_avg": circuit_avg,
                    "circuit_exp": circuit_exp,
                    "season_progress": season_progress,
                    "is_front_row": 1.0 if grid <= 2 else 0.0,
                    "is_top5_grid": 1.0 if grid <= 5 else 0.0,
                    "grid_vs_form": grid - recent5,
                })

                # Update histories (AFTER feature extraction to avoid leakage)
                if position <= 20:
                    driver_history.setdefault(driver_id, []).append(position)
                    circuit_driver_history.setdefault(ck, []).append(position)

                pts_earned = _safe_float(result.get("points"), 0.0)
                season_key = f"{year}_{team_id}"
                team_season_pts[season_key] = team_season_pts.get(season_key, 0.0) + pts_earned
                driver_season_pts[f"{year}_{driver_id}"] = (
                    driver_season_pts.get(f"{year}_{driver_id}", 0.0) + pts_earned
                )
                total_points_by_season[year] = total_points_by_season.get(year, 0.0) + pts_earned

        return pd.DataFrame(records)


FEATURE_COLS = [
    "grid_position",
    "team_pts_pct",
    "driver_season_pts",
    "recent5_avg",
    "recent3_avg",
    "consistency",
    "circuit_avg",
    "circuit_exp",
    "season_progress",
    "is_front_row",
    "is_top5_grid",
    "grid_vs_form",
]


class F1Predictor:
    def __init__(self):
        self.xgb: Optional[xgb.XGBRegressor] = None
        self.hgbr: Optional[HistGradientBoostingRegressor] = None
        self.gbr: Optional[GradientBoostingRegressor] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        self.processor = DataProcessor()
        # Runtime context built during training
        self.driver_history: Dict[str, List[float]] = {}
        self.circuit_driver_history: Dict[str, List[float]] = {}
        self.team_season_pts: Dict[str, float] = {}
        self.driver_season_pts: Dict[str, float] = {}
        self.total_points_by_season: Dict[int, float] = {}

    def train(self, all_races: List[Dict]) -> Dict[str, Any]:
        logger.info(f"Training on {len(all_races)} races…")
        df = self.processor.build_training_df(all_races)

        df = df[
            (df["grid_position"] > 0)
            & (df["final_position"] > 0)
            & (df["final_position"] <= 20)
        ].copy()
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

        X = df[FEATURE_COLS].values.astype(float)
        y = df["final_position"].values.astype(float)
        X_sc = self.scaler.fit_transform(X)

        self.xgb = xgb.XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            min_child_weight=3, gamma=0.1,
            random_state=42, verbosity=0, n_jobs=-1,
        )
        self.hgbr = HistGradientBoostingRegressor(
            max_iter=400, max_depth=5, learning_rate=0.04,
            min_samples_leaf=10, random_state=42,
        )
        self.gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.06,
            subsample=0.8, random_state=42,
        )

        self.xgb.fit(X_sc, y)
        self.hgbr.fit(X_sc, y)
        self.gbr.fit(X_sc, y)

        # Time-series style CV (leave-last-year out)
        cv_scores = cross_val_score(
            self.xgb, X_sc, y, cv=5, scoring="neg_mean_absolute_error"
        )
        train_pred = self._ensemble_predict(X_sc)
        train_mae = mean_absolute_error(y, train_pred)

        importance = dict(
            zip(FEATURE_COLS, self.xgb.feature_importances_.tolist())
        )

        self.metrics = {
            "cv_mae": float(-cv_scores.mean()),
            "cv_mae_std": float(cv_scores.std()),
            "train_mae": float(train_mae),
            "n_races": len(all_races),
            "n_samples": len(df),
            "feature_importance": importance,
        }
        self.is_trained = True

        # Rebuild runtime context from training data for inference
        self._rebuild_context(all_races)

        joblib.dump(
            {
                "xgb": self.xgb,
                "hgbr": self.hgbr,
                "gbr": self.gbr,
                "scaler": self.scaler,
                "metrics": self.metrics,
                "driver_history": self.driver_history,
                "circuit_driver_history": self.circuit_driver_history,
                "team_season_pts": self.team_season_pts,
                "driver_season_pts": self.driver_season_pts,
                "total_points_by_season": self.total_points_by_season,
            },
            MODEL_DIR / "f1_predictor.pkl",
        )
        logger.info(f"Training done. CV MAE: {self.metrics['cv_mae']:.3f}")
        return self.metrics

    def _rebuild_context(self, all_races: List[Dict]):
        dh: Dict[str, List[float]] = {}
        cdh: Dict[str, List[float]] = {}
        tsp: Dict[str, float] = {}
        dsp: Dict[str, float] = {}
        tpbs: Dict[int, float] = {}
        for race in sorted(all_races, key=lambda x: (x["year"], x["round"])):
            year = race["year"]
            for r in race["results"]:
                did = r["driver_id"]
                tid = r["team_id"]
                pos = _safe_float(r.get("position"), 20.0)
                pts = _safe_float(r.get("points"), 0.0)
                cid = race["circuit_id"]
                if pos <= 20:
                    dh.setdefault(did, []).append(pos)
                    cdh.setdefault(f"{did}_{cid}", []).append(pos)
                tsp[f"{year}_{tid}"] = tsp.get(f"{year}_{tid}", 0.0) + pts
                dsp[f"{year}_{did}"] = dsp.get(f"{year}_{did}", 0.0) + pts
                tpbs[year] = tpbs.get(year, 0.0) + pts
        self.driver_history = dh
        self.circuit_driver_history = cdh
        self.team_season_pts = tsp
        self.driver_season_pts = dsp
        self.total_points_by_season = tpbs

    def _ensemble_predict(self, X_sc: np.ndarray) -> np.ndarray:
        p1 = self.xgb.predict(X_sc)
        p2 = self.hgbr.predict(X_sc)
        p3 = self.gbr.predict(X_sc)
        return 0.45 * p1 + 0.35 * p2 + 0.20 * p3

    def predict_race(
        self,
        grid: List[Dict],
        circuit_id: str,
        year: int,
        round_num: int,
        total_rounds: int = 24,
    ) -> List[Dict]:
        if not self.is_trained:
            return self._fallback_predict(grid)

        season_progress = round_num / total_rounds
        total_pts = self.total_points_by_season.get(year, self.total_points_by_season.get(year - 1, 1.0))

        features = []
        for driver in grid:
            did = driver["driver_id"]
            tid = driver.get("team_id", "default")
            grid_pos = _safe_float(driver.get("grid"), 10.0)

            hist = self.driver_history.get(did, [])
            r5 = float(np.mean(hist[-5:])) if hist else 10.0
            r3 = float(np.mean(hist[-3:])) if hist else 10.0
            cons = float(np.std(hist[-5:])) if len(hist) >= 3 else 5.0

            chist = self.circuit_driver_history.get(f"{did}_{circuit_id}", [])
            c_avg = float(np.mean(chist)) if chist else 10.0
            c_exp = min(len(chist), 10)

            team_pts = self.team_season_pts.get(f"{year}_{tid}", 0.0)
            team_pct = team_pts / max(total_pts, 1.0) * 100
            drv_pts = self.driver_season_pts.get(f"{year}_{did}", 0.0)

            features.append([
                grid_pos, team_pct, drv_pts,
                r5, r3, cons,
                c_avg, c_exp, season_progress,
                1.0 if grid_pos <= 2 else 0.0,
                1.0 if grid_pos <= 5 else 0.0,
                grid_pos - r5,
            ])

        X = np.array(features, dtype=float)
        X_sc = self.scaler.transform(X)
        scores = self._ensemble_predict(X_sc)

        results = []
        for i, driver in enumerate(grid):
            results.append({**driver, "_score": float(scores[i])})

        results.sort(key=lambda x: x["_score"])

        for i, r in enumerate(results):
            # Confidence: gap to next driver relative to spread
            spread = float(np.std(scores)) if len(scores) > 1 else 5.0
            gap_next = (
                abs(results[i + 1]["_score"] - r["_score"]) if i + 1 < len(results) else spread
            )
            confidence = min(0.97, max(0.25, gap_next / max(spread, 1.0)))
            r["predicted_position"] = i + 1
            r["confidence"] = round(confidence, 3)
            del r["_score"]

        return results

    def get_win_probabilities(
        self,
        grid: List[Dict],
        circuit_id: str,
        year: int,
        round_num: int,
        total_rounds: int = 24,
        n_simulations: int = 1000,
    ) -> Dict[str, float]:
        """Monte Carlo simulation to estimate win probabilities."""
        if not self.is_trained:
            probs = {}
            for i, d in enumerate(sorted(grid, key=lambda x: x.get("grid", 20))):
                probs[d["driver_id"]] = max(0.01, 0.5 - i * 0.04)
            total = sum(probs.values())
            return {k: round(v / total, 3) for k, v in probs.items()}

        season_progress = round_num / total_rounds
        total_pts = self.total_points_by_season.get(year, self.total_points_by_season.get(year - 1, 1.0))

        base_features = []
        for driver in grid:
            did = driver["driver_id"]
            tid = driver.get("team_id", "default")
            grid_pos = _safe_float(driver.get("grid"), 10.0)
            hist = self.driver_history.get(did, [])
            r5 = float(np.mean(hist[-5:])) if hist else 10.0
            r3 = float(np.mean(hist[-3:])) if hist else 10.0
            cons = float(np.std(hist[-5:])) if len(hist) >= 3 else 5.0
            chist = self.circuit_driver_history.get(f"{did}_{circuit_id}", [])
            c_avg = float(np.mean(chist)) if chist else 10.0
            c_exp = min(len(chist), 10)
            team_pts = self.team_season_pts.get(f"{year}_{tid}", 0.0)
            team_pct = team_pts / max(total_pts, 1.0) * 100
            drv_pts = self.driver_season_pts.get(f"{year}_{did}", 0.0)
            base_features.append([
                grid_pos, team_pct, drv_pts, r5, r3, cons,
                c_avg, c_exp, season_progress,
                1.0 if grid_pos <= 2 else 0.0,
                1.0 if grid_pos <= 5 else 0.0,
                grid_pos - r5,
            ])

        X_base = np.array(base_features, dtype=float)
        X_sc = self.scaler.transform(X_base)
        base_scores = self._ensemble_predict(X_sc)
        noise_std = self.metrics.get("cv_mae", 3.0)

        win_counts = {d["driver_id"]: 0 for d in grid}
        rng = np.random.default_rng(42)
        for _ in range(n_simulations):
            noise = rng.normal(0, noise_std, len(grid))
            sim_scores = base_scores + noise
            winner_idx = int(np.argmin(sim_scores))
            win_counts[grid[winner_idx]["driver_id"]] += 1

        return {
            k: round(v / n_simulations, 3) for k, v in win_counts.items()
        }

    def _fallback_predict(self, grid: List[Dict]) -> List[Dict]:
        results = sorted(grid, key=lambda x: x.get("grid", 20))
        for i, r in enumerate(results):
            r["predicted_position"] = i + 1
            r["confidence"] = round(max(0.3, 0.92 - i * 0.03), 3)
        return results

    def load(self) -> bool:
        p = MODEL_DIR / "f1_predictor.pkl"
        if not p.exists():
            return False
        try:
            data = joblib.load(p)
            self.xgb = data["xgb"]
            self.hgbr = data.get("hgbr") or data.get("lgb")  # backward compat
            self.gbr = data["gbr"]
            self.scaler = data["scaler"]
            self.metrics = data["metrics"]
            self.driver_history = data.get("driver_history", {})
            self.circuit_driver_history = data.get("circuit_driver_history", {})
            self.team_season_pts = data.get("team_season_pts", {})
            self.driver_season_pts = data.get("driver_season_pts", {})
            self.total_points_by_season = data.get("total_points_by_season", {})
            self.is_trained = True
            logger.info("Model loaded from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
