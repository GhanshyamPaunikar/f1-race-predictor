"""
F1 Race Predictor — FastAPI backend
Serves predictions, standings, and model status.
Trains on historical data from Ergast/Jolpica API (2018–2025).
"""
import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import data_fetcher as df
from predictor import F1Predictor, TEAM_COLORS, _safe_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

TRAINING_YEARS = list(range(2018, 2026))  # 2018–2025
CURRENT_YEAR = 2026

predictor = F1Predictor()
training_status = {"state": "idle", "progress": 0, "message": ""}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _team_color(team_id: str) -> str:
    for key, color in TEAM_COLORS.items():
        if key in team_id.lower():
            return color
    return TEAM_COLORS["default"]


def _parse_all_races(races_by_year: Dict[int, list]) -> List[Dict]:
    """Flatten API race data into the format the predictor expects."""
    all_races = []
    for year, races in races_by_year.items():
        total_rounds = len(races)
        for race in races:
            results = []
            for r in race.get("Results", []):
                pos_raw = r.get("position", "20")
                try:
                    pos = int(pos_raw)
                except ValueError:
                    pos = 20
                grid_raw = r.get("grid", "0")
                try:
                    grid = int(grid_raw)
                    if grid == 0:
                        grid = 20  # pit lane starts
                except ValueError:
                    grid = 20
                results.append({
                    "driver_id": r["Driver"]["driverId"],
                    "team_id": r["Constructor"]["constructorId"],
                    "position": pos,
                    "grid": grid,
                    "points": _safe_float(r.get("points"), 0.0),
                    "status": r.get("status", ""),
                })
            all_races.append({
                "year": year,
                "round": int(race.get("round", 0)),
                "circuit_id": race["Circuit"]["circuitId"],
                "race_name": race.get("raceName", ""),
                "date": race.get("date", ""),
                "total_rounds": total_rounds,
                "results": results,
            })
    return all_races


async def _train_background():
    global training_status
    training_status = {"state": "fetching", "progress": 5, "message": "Fetching historical race data…"}
    try:
        races_by_year: Dict[int, list] = {}
        for i, year in enumerate(TRAINING_YEARS):
            training_status["progress"] = 5 + int(60 * i / len(TRAINING_YEARS))
            training_status["message"] = f"Loading {year} season data…"
            races = await asyncio.to_thread(df.get_race_results, year)
            if races:
                races_by_year[year] = races
            await asyncio.sleep(0.1)

        training_status = {"state": "training", "progress": 70, "message": "Training ML ensemble…"}
        all_races = _parse_all_races(races_by_year)
        metrics = await asyncio.to_thread(predictor.train, all_races)

        training_status = {
            "state": "done",
            "progress": 100,
            "message": f"Training complete. CV MAE: {metrics['cv_mae']:.2f} positions",
            "metrics": metrics,
        }
        logger.info("Background training finished.")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        training_status = {"state": "error", "progress": 0, "message": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load a pre-trained model; train in background if not found
    loaded = predictor.load()
    if not loaded:
        logger.info("No saved model found — starting background training…")
        asyncio.create_task(_train_background())
    else:
        training_status["state"] = "done"
        training_status["progress"] = 100
        training_status["message"] = "Model loaded from disk."
        training_status["metrics"] = predictor.metrics
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="F1 Race Predictor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def status():
    return {
        "model_trained": predictor.is_trained,
        "training": training_status,
        "metrics": predictor.metrics if predictor.is_trained else None,
    }


@app.post("/api/train")
async def trigger_train(background_tasks: BackgroundTasks):
    if training_status.get("state") == "fetching" or training_status.get("state") == "training":
        return {"message": "Training already in progress", "status": training_status}
    background_tasks.add_task(_train_background)
    return {"message": "Training started"}


@app.get("/api/schedule/{year}")
async def race_schedule(year: int):
    races = await asyncio.to_thread(df.get_schedule, year)
    today = date.today().isoformat()
    result = []
    for race in races:
        race_date = race.get("date", "")
        result.append({
            "round": int(race.get("round", 0)),
            "name": race.get("raceName", ""),
            "circuit": race["Circuit"]["circuitName"],
            "circuit_id": race["Circuit"]["circuitId"],
            "country": race["Circuit"]["Location"]["country"],
            "locality": race["Circuit"]["Location"]["locality"],
            "date": race_date,
            "is_past": race_date < today,
            "is_upcoming": race_date >= today,
        })
    return result


@app.get("/api/standings/drivers/{year}")
async def driver_standings(year: int):
    standings = await asyncio.to_thread(df.get_driver_standings, year)
    return [
        {
            "position": int(s.get("position", 0)),
            "driver_id": s["Driver"]["driverId"],
            "name": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
            "team": s["Constructors"][0]["name"] if s.get("Constructors") else "—",
            "team_id": s["Constructors"][0]["constructorId"] if s.get("Constructors") else "default",
            "points": float(s.get("points", 0)),
            "wins": int(s.get("wins", 0)),
            "team_color": _team_color(
                s["Constructors"][0]["constructorId"] if s.get("Constructors") else ""
            ),
        }
        for s in standings
    ]


@app.get("/api/standings/constructors/{year}")
async def constructor_standings(year: int):
    standings = await asyncio.to_thread(df.get_constructor_standings, year)
    return [
        {
            "position": int(s.get("position", 0)),
            "team_id": s["Constructor"]["constructorId"],
            "name": s["Constructor"]["name"],
            "points": float(s.get("points", 0)),
            "wins": int(s.get("wins", 0)),
            "team_color": _team_color(s["Constructor"]["constructorId"]),
        }
        for s in standings
    ]


@app.get("/api/predict/{year}/{round_num}")
async def predict_race(year: int, round_num: int):
    schedule = await asyncio.to_thread(df.get_schedule, year)
    race_info = next((r for r in schedule if int(r.get("round", 0)) == round_num), None)
    if not race_info:
        raise HTTPException(404, f"Race {year} R{round_num} not found")

    circuit_id = race_info["Circuit"]["circuitId"]
    total_rounds = len(schedule)

    # Try to get qualifying results for the grid
    quali_results = await asyncio.to_thread(df.get_round_qualifying, year, round_num)
    race_results = await asyncio.to_thread(df.get_round_result, year, round_num)

    # Build grid from qualifying; fall back to driver standings order
    grid: List[Dict] = []
    if quali_results:
        for q in quali_results:
            grid.append({
                "driver_id": q["Driver"]["driverId"],
                "name": f"{q['Driver']['givenName']} {q['Driver']['familyName']}",
                "team": q["Constructor"]["name"],
                "team_id": q["Constructor"]["constructorId"],
                "team_color": _team_color(q["Constructor"]["constructorId"]),
                "grid": int(q.get("position", 20)),
                "number": q["Driver"].get("permanentNumber", "?"),
                "nationality": q["Driver"].get("nationality", ""),
            })
    else:
        # No qualifying data — use season standings as grid proxy
        driver_standings = await asyncio.to_thread(df.get_driver_standings, year)
        for i, s in enumerate(driver_standings[:20]):
            team_id = s["Constructors"][0]["constructorId"] if s.get("Constructors") else "default"
            grid.append({
                "driver_id": s["Driver"]["driverId"],
                "name": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                "team": s["Constructors"][0]["name"] if s.get("Constructors") else "—",
                "team_id": team_id,
                "team_color": _team_color(team_id),
                "grid": i + 1,
                "number": s["Driver"].get("permanentNumber", "?"),
                "nationality": s["Driver"].get("nationality", ""),
            })

    if not grid:
        raise HTTPException(400, "No grid data available for this race")

    # ML predictions
    predictions = predictor.predict_race(grid, circuit_id, year, round_num, total_rounds)
    win_probs = predictor.get_win_probabilities(grid, circuit_id, year, round_num, total_rounds)

    for p in predictions:
        p["win_probability"] = win_probs.get(p["driver_id"], 0.0)

    # If race already happened, include actual results
    actual: Dict[str, int] = {}
    if race_results:
        for r in race_results:
            did = r["Driver"]["driverId"]
            try:
                actual[did] = int(r["position"])
            except (ValueError, KeyError):
                pass

    return {
        "race": {
            "year": year,
            "round": round_num,
            "name": race_info.get("raceName", ""),
            "circuit": race_info["Circuit"]["circuitName"],
            "circuit_id": circuit_id,
            "country": race_info["Circuit"]["Location"]["country"],
            "locality": race_info["Circuit"]["Location"]["locality"],
            "date": race_info.get("date", ""),
            "has_result": bool(race_results),
        },
        "predictions": predictions,
        "actual_results": actual,
        "model_trained": predictor.is_trained,
        "feature_importance": predictor.metrics.get("feature_importance", {}),
    }


@app.get("/api/model/accuracy")
async def model_accuracy():
    if not predictor.is_trained:
        return {"trained": False}
    return {"trained": True, **predictor.metrics}


@app.get("/api/history/{year}/{round_num}")
async def race_history(year: int, round_num: int):
    results = await asyncio.to_thread(df.get_round_result, year, round_num)
    return [
        {
            "position": r.get("position"),
            "driver": f"{r['Driver']['givenName']} {r['Driver']['familyName']}",
            "driver_id": r["Driver"]["driverId"],
            "team": r["Constructor"]["name"],
            "team_color": _team_color(r["Constructor"]["constructorId"]),
            "grid": r.get("grid"),
            "points": r.get("points"),
            "status": r.get("status"),
            "laps": r.get("laps"),
        }
        for r in results
    ]


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def root():
        return {"message": "Frontend not found. Run from the project root."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
