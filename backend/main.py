"""
F1 Race Predictor — FastAPI backend (v2).

Serves predictions, standings, model status, per-prediction explanations,
head-to-head comparisons and historical prediction accuracy.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import data_fetcher as df
import practice_features as pf
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from predictor import FEATURE_LABELS, TEAM_COLORS, F1Predictor, _safe_float

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

TRAINING_YEARS = list(range(2018, 2026))  # 2018–2025
CURRENT_YEAR = 2026


@dataclass
class AppState:
    predictor: F1Predictor = field(default_factory=F1Predictor)
    training_status: dict[str, Any] = field(default_factory=lambda: {"state": "idle", "progress": 0, "message": ""})
    training_task: asyncio.Task | None = None
    # driver_id → 3-letter FastF1 code; built during training, used at inference
    id_to_code: dict[str, str] = field(default_factory=dict)


state = AppState()

# ── Helpers ────────────────────────────────────────────────────────────


def _team_color(team_id: str) -> str:
    tid = (team_id or "").lower()
    for key, color in TEAM_COLORS.items():
        if key in tid:
            return color
    return TEAM_COLORS["default"]


def _parse_all_races(races_by_year: dict[int, list]) -> list[dict]:
    out: list[dict] = []
    for year, races in races_by_year.items():
        total_rounds = len(races)
        for race in races:
            results = []
            for r in race.get("Results", []):
                try:
                    pos = int(r.get("position", "20"))
                except (ValueError, TypeError):
                    pos = 20
                try:
                    grid = int(r.get("grid", "0")) or 20
                except (ValueError, TypeError):
                    grid = 20
                results.append({
                    "driver_id": r["Driver"]["driverId"],
                    "team_id": r["Constructor"]["constructorId"],
                    "position": pos,
                    "grid": grid,
                    "points": _safe_float(r.get("points"), 0.0),
                    "status": r.get("status", ""),
                })
            out.append({
                "year": year,
                "round": int(race.get("round", 0)),
                "circuit_id": race["Circuit"]["circuitId"],
                "race_name": race.get("raceName", ""),
                "date": race.get("date", ""),
                "total_rounds": total_rounds,
                "results": results,
            })
    return out


async def _train_background() -> None:
    state.training_status = {"state": "fetching", "progress": 5, "message": "Fetching historical race data…"}
    try:
        races_by_year: dict[int, list] = {}
        for i, year in enumerate(TRAINING_YEARS):
            state.training_status["progress"] = 5 + int(50 * i / len(TRAINING_YEARS))
            state.training_status["message"] = f"Loading {year} season data…"
            races = await asyncio.to_thread(df.get_race_results, year)
            if races:
                races_by_year[year] = races
            await asyncio.sleep(0.05)

        state.training_status = {"state": "fetching", "progress": 55, "message": "Fetching circuit weather…"}

        def _weather_progress(done: int, total: int) -> None:
            pct = 55 + int(10 * done / max(total, 1))
            state.training_status["progress"] = pct
            state.training_status["message"] = f"Fetching circuit weather… {done}/{total}"

        weather_lookup = await asyncio.to_thread(
            df.build_weather_lookup, races_by_year, _weather_progress,
        )

        state.training_status = {"state": "fetching", "progress": 65, "message": "Fetching qualifying timings…"}

        def _quali_progress(done: int, total: int) -> None:
            pct = 65 + int(10 * done / max(total, 1))
            state.training_status["progress"] = pct
            state.training_status["message"] = f"Fetching qualifying timings… {done}/{total}"

        quali_gap_lookup = await asyncio.to_thread(
            df.build_quali_gap_lookup, races_by_year, _quali_progress,
        )

        state.training_status = {"state": "fetching", "progress": 75, "message": "Fetching practice sessions (FastF1)…"}

        def _practice_progress(done: int, total: int) -> None:
            pct = 75 + int(15 * done / max(total, 1))
            state.training_status["progress"] = pct
            state.training_status["message"] = f"Fetching practice sessions… {done}/{total}"

        practice_lookup = await asyncio.to_thread(
            pf.build_practice_lookup, races_by_year, _practice_progress,
        )
        # Build {driver_id: code} so the predictor can look up practice features
        code_to_id = pf.build_code_to_id_mapping(races_by_year)
        state.id_to_code = {did: code for code, did in code_to_id.items()}

        state.training_status = {"state": "training", "progress": 92, "message": "Training ML ensemble…"}
        all_races = _parse_all_races(races_by_year)
        metrics = await asyncio.to_thread(
            state.predictor.train,
            all_races, weather_lookup, quali_gap_lookup, practice_lookup, state.id_to_code,
        )

        state.training_status = {
            "state": "done",
            "progress": 100,
            "message": (
                f"CV MAE: {metrics['cv_mae']:.2f} pos · "
                f"Podium hits: {metrics['podium_hit_rate']*100:.0f}%"
            ),
            "metrics": metrics,
        }
        logger.info("Background training finished.")
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        state.training_status = {"state": "error", "progress": 0, "message": str(e)}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if state.predictor.load():
        state.training_status = {
            "state": "done", "progress": 100,
            "message": "Model loaded from disk.",
            "metrics": state.predictor.metrics,
        }
    else:
        logger.info("No saved model found — starting background training…")
        state.training_task = asyncio.create_task(_train_background())
    yield


# ── App ────────────────────────────────────────────────────────────────

app = FastAPI(title="F1 Race Predictor", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ── Routes ─────────────────────────────────────────────────────────────


@app.get("/api/status")
async def status() -> dict:
    metrics = None
    if state.predictor.is_trained:
        # Include the friendly feature-name dictionary so the frontend can
        # always render labels without keeping a duplicate copy in sync.
        metrics = {**state.predictor.metrics, "feature_labels": FEATURE_LABELS}
    return {
        "model_trained": state.predictor.is_trained,
        "training": state.training_status,
        "metrics": metrics,
    }


@app.post("/api/train")
async def trigger_train(background_tasks: BackgroundTasks) -> dict:
    if state.training_status.get("state") in {"fetching", "training"}:
        return {"message": "Training already in progress", "status": state.training_status}
    background_tasks.add_task(_train_background)
    return {"message": "Training started"}


@app.get("/api/schedule/{year}")
async def race_schedule(year: int) -> list[dict]:
    races = await asyncio.to_thread(df.get_schedule, year)
    if not races:
        raise HTTPException(404, f"No schedule available for {year}")
    today = date.today().isoformat()
    return [
        {
            "round": int(r.get("round", 0)),
            "name": r.get("raceName", ""),
            "circuit": r["Circuit"]["circuitName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "country": r["Circuit"]["Location"]["country"],
            "locality": r["Circuit"]["Location"]["locality"],
            "date": r.get("date", ""),
            "is_past": r.get("date", "") < today,
            "is_upcoming": r.get("date", "") >= today,
        }
        for r in races
    ]


@app.get("/api/standings/drivers/{year}")
async def driver_standings(year: int) -> list[dict]:
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
async def constructor_standings(year: int) -> list[dict]:
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


def _practice_by_driver_id(practice_by_code: dict[str, dict]) -> dict[str, dict]:
    """Convert {3-letter code: feats} → {driver_id: feats} using the cached mapping."""
    if not practice_by_code:
        return {}
    out: dict[str, dict] = {}
    for code, feats in practice_by_code.items():
        did = next(
            (d for d, c in state.id_to_code.items() if c == code.upper()),
            None,
        )
        if did:
            out[did] = feats
    return out


async def _build_grid(year: int, round_num: int, schedule: list) -> tuple[list[dict], dict]:
    race_info = next((r for r in schedule if int(r.get("round", 0)) == round_num), None)
    if not race_info:
        raise HTTPException(404, f"Race {year} R{round_num} not found")
    quali_results = await asyncio.to_thread(df.get_round_qualifying, year, round_num)
    grid: list[dict] = []
    if quali_results:
        for q in quali_results:
            tid = q["Constructor"]["constructorId"]
            did = q["Driver"]["driverId"]
            grid.append({
                "driver_id": did,
                "name": f"{q['Driver']['givenName']} {q['Driver']['familyName']}",
                "team": q["Constructor"]["name"],
                "team_id": tid,
                "team_color": _team_color(tid),
                "grid": int(q.get("position", 20)),
                "number": q["Driver"].get("permanentNumber", "?"),
                "nationality": q["Driver"].get("nationality", ""),
                "code": state.id_to_code.get(did, q["Driver"].get("code", "")).upper(),
            })
    else:
        standings = await asyncio.to_thread(df.get_driver_standings, year)
        for i, s in enumerate(standings[:20]):
            tid = s["Constructors"][0]["constructorId"] if s.get("Constructors") else "default"
            did = s["Driver"]["driverId"]
            grid.append({
                "driver_id": did,
                "name": f"{s['Driver']['givenName']} {s['Driver']['familyName']}",
                "team": s["Constructors"][0]["name"] if s.get("Constructors") else "—",
                "team_id": tid,
                "team_color": _team_color(tid),
                "grid": i + 1,
                "number": s["Driver"].get("permanentNumber", "?"),
                "nationality": s["Driver"].get("nationality", ""),
                "code": state.id_to_code.get(did, s["Driver"].get("code", "")).upper(),
            })
    return grid, race_info


@app.get("/api/predict/{year}/{round_num}")
async def predict_race(year: int, round_num: int) -> dict:
    schedule = await asyncio.to_thread(df.get_schedule, year)
    grid, race_info = await _build_grid(year, round_num, schedule)
    if not grid:
        raise HTTPException(400, "No grid data available for this race")

    circuit_id = race_info["Circuit"]["circuitId"]
    total_rounds = len(schedule)
    weather = await asyncio.to_thread(df.get_weather, circuit_id, race_info.get("date", ""))
    quali_gaps = await asyncio.to_thread(df.get_qualifying_gaps, year, round_num)
    practice_by_code = await asyncio.to_thread(pf.fetch_practice_features, year, round_num)
    practice = _practice_by_driver_id(practice_by_code)
    race_results = await asyncio.to_thread(df.get_round_result, year, round_num)

    predictions = state.predictor.predict_race(
        grid, circuit_id, year, round_num, total_rounds,
        weather=weather, quali_gaps=quali_gaps, practice=practice,
    )
    win_probs = state.predictor.get_win_probabilities(
        grid, circuit_id, year, round_num, total_rounds,
        weather=weather, quali_gaps=quali_gaps, practice=practice,
    )
    for p in predictions:
        p["win_probability"] = win_probs.get(p["driver_id"], 0.0)

    actual: dict[str, int] = {}
    if race_results:
        for r in race_results:
            with suppress(ValueError, KeyError):
                actual[r["Driver"]["driverId"]] = int(r["position"])

    return {
        "race": {
            "year": year, "round": round_num,
            "name": race_info.get("raceName", ""),
            "circuit": race_info["Circuit"]["circuitName"],
            "circuit_id": circuit_id,
            "country": race_info["Circuit"]["Location"]["country"],
            "locality": race_info["Circuit"]["Location"]["locality"],
            "date": race_info.get("date", ""),
            "has_result": bool(race_results),
            "weather": weather,
        },
        "predictions": predictions,
        "actual_results": actual,
        "model_trained": state.predictor.is_trained,
        "feature_importance": state.predictor.metrics.get("feature_importance", {}),
    }


@app.get("/api/explain/{year}/{round_num}/{driver_id}")
async def explain_prediction(year: int, round_num: int, driver_id: str) -> dict:
    if not state.predictor.is_trained:
        raise HTTPException(409, "Model not trained yet")
    schedule = await asyncio.to_thread(df.get_schedule, year)
    grid, race_info = await _build_grid(year, round_num, schedule)
    circuit_id = race_info["Circuit"]["circuitId"]
    weather = await asyncio.to_thread(df.get_weather, circuit_id, race_info.get("date", ""))
    quali_gaps = await asyncio.to_thread(df.get_qualifying_gaps, year, round_num)
    practice_by_code = await asyncio.to_thread(pf.fetch_practice_features, year, round_num)
    practice = _practice_by_driver_id(practice_by_code)
    contributions = state.predictor.explain_driver(
        grid, driver_id, circuit_id, year, round_num, len(schedule),
        weather=weather, quali_gaps=quali_gaps, practice=practice,
    )
    if not contributions:
        raise HTTPException(404, f"Driver {driver_id} not on grid")
    return {"driver_id": driver_id, "contributions": contributions}


@app.get("/api/compare/{year}/{round_num}")
async def compare_drivers(year: int, round_num: int, drivers: str) -> dict:
    """Head-to-head: drivers param is comma-separated driver_ids."""
    ids = [d.strip() for d in drivers.split(",") if d.strip()]
    if len(ids) < 2:
        raise HTTPException(400, "Provide at least two driver_ids, comma-separated")

    schedule = await asyncio.to_thread(df.get_schedule, year)
    grid, race_info = await _build_grid(year, round_num, schedule)
    circuit_id = race_info["Circuit"]["circuitId"]
    weather = await asyncio.to_thread(df.get_weather, circuit_id, race_info.get("date", ""))
    quali_gaps = await asyncio.to_thread(df.get_qualifying_gaps, year, round_num)
    practice_by_code = await asyncio.to_thread(pf.fetch_practice_features, year, round_num)
    practice = _practice_by_driver_id(practice_by_code)
    preds = state.predictor.predict_race(
        grid, circuit_id, year, round_num, len(schedule),
        weather=weather, quali_gaps=quali_gaps, practice=practice,
    )
    win_probs = state.predictor.get_win_probabilities(
        grid, circuit_id, year, round_num, len(schedule),
        weather=weather, quali_gaps=quali_gaps, practice=practice,
    )

    by_id = {p["driver_id"]: p for p in preds}
    matched = []
    for did in ids:
        p = by_id.get(did)
        if not p:
            continue
        hist = state.predictor.ctx.driver_history.get(did, [])
        chist = state.predictor.ctx.circuit_driver_history.get(f"{did}_{circuit_id}", [])
        matched.append({
            **p,
            "win_probability": win_probs.get(did, 0.0),
            "form_last5": hist[-5:],
            "circuit_history": chist,
            "season_avg_finish": (sum(hist) / len(hist)) if hist else None,
        })
    if not matched:
        raise HTTPException(404, "None of the requested drivers found on grid")
    return {"race": race_info.get("raceName", ""), "drivers": matched}


@app.get("/api/predictions/history/{year}")
async def predictions_history(year: int) -> dict:
    """Back-tests the trained model against every completed race in `year` and
    reports per-race podium hit rate and exact-position hits."""
    if not state.predictor.is_trained:
        raise HTTPException(409, "Model not trained yet")
    schedule = await asyncio.to_thread(df.get_schedule, year)
    today = date.today().isoformat()

    races = []
    cum_p1, cum_p3, cum_top10, total = 0, 0, 0, 0
    for r in schedule:
        if r.get("date", "") >= today:
            continue
        rnd = int(r.get("round", 0))
        results = await asyncio.to_thread(df.get_round_result, year, rnd)
        if not results:
            continue
        try:
            grid, info = await _build_grid(year, rnd, schedule)
        except HTTPException:
            continue
        weather = await asyncio.to_thread(df.get_weather, info["Circuit"]["circuitId"], r.get("date", ""))
        quali_gaps = await asyncio.to_thread(df.get_qualifying_gaps, year, rnd)
        practice_by_code = await asyncio.to_thread(pf.fetch_practice_features, year, rnd)
        practice = _practice_by_driver_id(practice_by_code)
        preds = state.predictor.predict_race(
            grid, info["Circuit"]["circuitId"], year, rnd, len(schedule),
            weather=weather, quali_gaps=quali_gaps, practice=practice,
        )
        actual = {res["Driver"]["driverId"]: int(res["position"]) for res in results
                  if str(res.get("position", "")).isdigit()}
        p1_pred = next((p for p in preds if p["predicted_position"] == 1), None)
        p1_hit = bool(p1_pred and actual.get(p1_pred["driver_id"]) == 1)
        podium_pred = {p["driver_id"] for p in preds if p["predicted_position"] <= 3}
        podium_actual = {d for d, pos in actual.items() if pos <= 3}
        podium_hits = len(podium_pred & podium_actual)
        top10_pred = {p["driver_id"] for p in preds if p["predicted_position"] <= 10}
        top10_actual = {d for d, pos in actual.items() if pos <= 10}
        top10_hits = len(top10_pred & top10_actual)

        cum_p1 += int(p1_hit)
        cum_p3 += podium_hits
        cum_top10 += top10_hits
        total += 1
        races.append({
            "round": rnd,
            "name": r.get("raceName", ""),
            "country": r["Circuit"]["Location"]["country"],
            "date": r.get("date", ""),
            "winner_hit": p1_hit,
            "podium_hits": podium_hits,
            "top10_hits": top10_hits,
            "predicted_winner": p1_pred["name"] if p1_pred else None,
            "actual_winner": next((res["Driver"]["givenName"] + " " + res["Driver"]["familyName"]
                                   for res in results if str(res.get("position")) == "1"), None),
        })

    return {
        "year": year,
        "races": races,
        "summary": {
            "races_evaluated": total,
            "winner_hit_rate": round(cum_p1 / total, 3) if total else 0.0,
            "podium_hit_rate": round(cum_p3 / max(total * 3, 1), 3),
            "top10_hit_rate": round(cum_top10 / max(total * 10, 1), 3),
        },
    }


@app.get("/api/driver/{driver_id}")
async def driver_detail(driver_id: str) -> dict:
    """Aggregated career stats for one driver from the trained model context."""
    if not state.predictor.is_trained:
        raise HTTPException(409, "Model not trained yet")
    ctx = state.predictor.ctx
    hist = ctx.driver_history.get(driver_id, [])
    dnfs = ctx.driver_dnf_history.get(driver_id, [])
    # Keys look like "max_verstappen_monza" — driver_ids themselves contain
    # underscores, so naive split breaks. Match by prefix instead.
    by_circuit: dict[str, list[float]] = {}
    prefix = f"{driver_id}_"
    for k, v in ctx.circuit_driver_history.items():
        if k.startswith(prefix):
            cid = k[len(prefix):]
            if cid:
                by_circuit[cid] = v
    return {
        "driver_id": driver_id,
        "races": len(hist),
        "avg_finish": (sum(hist) / len(hist)) if hist else None,
        "best_finish": min(hist) if hist else None,
        "dnf_rate": (sum(dnfs) / len(dnfs)) if dnfs else 0.0,
        "form_last5": hist[-5:],
        "form_last10": hist[-10:],
        "by_circuit": {cid: {"avg": sum(v) / len(v), "races": len(v)} for cid, v in by_circuit.items()},
    }


@app.get("/api/model/accuracy")
async def model_accuracy() -> dict:
    if not state.predictor.is_trained:
        return {"trained": False}
    return {"trained": True, "feature_labels": FEATURE_LABELS, **state.predictor.metrics}


@app.get("/api/history/{year}/{round_num}")
async def race_history(year: int, round_num: int) -> list[dict]:
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


# ── Static frontend ────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def root() -> dict:
        return {"message": "Frontend not found. Run from the project root."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
