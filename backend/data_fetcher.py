"""Local-cached HTTP client for the Jolpica/Ergast F1 API and Open-Meteo weather."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://api.jolpi.ca/ergast/f1"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# Lat/lon for circuit weather lookup. Add as needed; predictor falls back to
# neutral defaults when missing.
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "albert_park":  (-37.8497,  144.9680),
    "bahrain":      ( 26.0325,   50.5106),
    "jeddah":       ( 21.6319,   39.1044),
    "shanghai":     ( 31.3389,  121.2199),
    "suzuka":       ( 34.8431,  136.5414),
    "miami":        ( 25.9581,  -80.2389),
    "imola":        ( 44.3439,   11.7167),
    "monaco":       ( 43.7347,    7.4206),
    "villeneuve":   ( 45.5000,  -73.5228),
    "catalunya":    ( 41.5700,    2.2611),
    "red_bull_ring":( 47.2197,   14.7647),
    "silverstone":  ( 52.0786,   -1.0169),
    "hungaroring":  ( 47.5789,   19.2486),
    "spa":          ( 50.4372,    5.9714),
    "zandvoort":    ( 52.3888,    4.5409),
    "monza":        ( 45.6156,    9.2811),
    "baku":         ( 40.3725,   49.8533),
    "marina_bay":   (  1.2914,  103.8645),
    "americas":     ( 30.1328,  -97.6411),
    "rodriguez":    ( 19.4042,  -99.0907),
    "interlagos":   (-23.7036,  -46.6997),
    "vegas":        ( 36.1147, -115.1728),
    "losail":       ( 25.4900,   51.4542),
    "yas_marina":   ( 24.4672,   54.6031),
}


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _fetch_with_cache(url: str, key: str, ttl: int = 3600) -> dict:
    cf = _cache_path(key)
    if cf.exists() and (time.time() - cf.stat().st_mtime) < ttl:
        try:
            return json.loads(cf.read_text())
        except Exception:
            pass
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "F1RacePredictor/2.0"})
            resp.raise_for_status()
            data = resp.json()
        cf.write_text(json.dumps(data))
        return data
    except Exception as e:
        logger.warning("API fetch failed for %s: %s", url, e)
        if cf.exists():
            try:
                return json.loads(cf.read_text())
            except Exception:
                return {}
        return {}


def _paginated(year: int, endpoint: str, key_prefix: str) -> list:
    out: list = []
    offset, limit = 0, 100
    while True:
        url = f"{BASE_URL}/{year}/{endpoint}.json?limit={limit}&offset={offset}"
        data = _fetch_with_cache(url, f"{key_prefix}_{year}_{offset}", ttl=86400)
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            break
        out.extend(races)
        total = int(data.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break
    return out


def get_race_results(year: int) -> list:
    return _paginated(year, "results", "results")


def get_qualifying(year: int) -> list:
    return _paginated(year, "qualifying", "quali")


def get_schedule(year: int) -> list:
    url = f"{BASE_URL}/{year}.json"
    data = _fetch_with_cache(url, f"schedule_{year}", ttl=86400)
    return data.get("MRData", {}).get("RaceTable", {}).get("Races", [])


def get_driver_standings(year: int, round_num: int | None = None) -> list:
    if round_num:
        url = f"{BASE_URL}/{year}/{round_num}/driverStandings.json"
        key = f"driver_standings_{year}_{round_num}"
    else:
        url = f"{BASE_URL}/{year}/driverStandings.json"
        key = f"driver_standings_{year}"
    data = _fetch_with_cache(url, key, ttl=3600)
    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("DriverStandings", []) if lists else []


def get_constructor_standings(year: int, round_num: int | None = None) -> list:
    if round_num:
        url = f"{BASE_URL}/{year}/{round_num}/constructorStandings.json"
        key = f"ctor_standings_{year}_{round_num}"
    else:
        url = f"{BASE_URL}/{year}/constructorStandings.json"
        key = f"ctor_standings_{year}"
    data = _fetch_with_cache(url, key, ttl=3600)
    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("ConstructorStandings", []) if lists else []


def get_round_result(year: int, round_num: int) -> list:
    url = f"{BASE_URL}/{year}/{round_num}/results.json?limit=25"
    data = _fetch_with_cache(url, f"result_{year}_{round_num}", ttl=86400)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("Results", []) if races else []


def get_round_qualifying(year: int, round_num: int) -> list:
    url = f"{BASE_URL}/{year}/{round_num}/qualifying.json?limit=25"
    data = _fetch_with_cache(url, f"quali_{year}_{round_num}", ttl=86400)
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return races[0].get("QualifyingResults", []) if races else []


def _quali_time_to_seconds(t: str) -> float | None:
    """Parse Ergast quali time string like '1:18.475' or '58.231' to seconds."""
    if not t:
        return None
    try:
        if ":" in t:
            mins, rest = t.split(":", 1)
            return int(mins) * 60 + float(rest)
        return float(t)
    except (ValueError, TypeError):
        return None


def get_qualifying_gaps(year: int, round_num: int) -> dict[str, float]:
    """Returns {driver_id: gap_to_pole_seconds}. Uses best of Q1/Q2/Q3 per driver."""
    quali = get_round_qualifying(year, round_num)
    if not quali:
        return {}
    best_times: dict[str, float] = {}
    for q in quali:
        did = q.get("Driver", {}).get("driverId", "")
        if not did:
            continue
        for col in ("Q3", "Q2", "Q1"):
            secs = _quali_time_to_seconds(q.get(col, ""))
            if secs is not None:
                best_times[did] = secs
                break
    if not best_times:
        return {}
    pole = min(best_times.values())
    return {did: round(t - pole, 3) for did, t in best_times.items()}


def build_quali_gap_lookup(
    races_by_year: dict[int, list],
    progress_cb=None,
) -> dict[tuple, dict[str, float]]:
    """For training: {(year, round): {driver_id: gap_to_pole}} for all races."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = [
        (year, int(r.get("round", 0)))
        for year, races in races_by_year.items()
        for r in races
        if str(r.get("round", "")).isdigit()
    ]
    out: dict[tuple, dict[str, float]] = {}
    total = len(jobs)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(get_qualifying_gaps, y, r): (y, r) for y, r in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                out[key] = fut.result()
            except Exception:
                out[key] = {}
            if progress_cb:
                progress_cb(i, total)
    return out


def get_driver_info(year: int) -> list:
    url = f"{BASE_URL}/{year}/drivers.json?limit=100"
    data = _fetch_with_cache(url, f"drivers_{year}", ttl=86400)
    return data.get("MRData", {}).get("DriverTable", {}).get("Drivers", [])


# ── Weather ─────────────────────────────────────────────────────────────


def get_weather(circuit_id: str, race_date: str) -> dict[str, float]:
    """Returns {air_temp_c, precip_mm} for a circuit on a given race date.
    Uses Open-Meteo historical archive; cached forever (the past doesn't change)."""
    coords = CIRCUIT_COORDS.get(circuit_id)
    if not coords or not race_date:
        return {"air_temp_c": 22.0, "precip_mm": 0.0}
    lat, lon = coords
    url = (
        f"{WEATHER_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={race_date}&end_date={race_date}"
        f"&daily=temperature_2m_max,precipitation_sum&timezone=UTC"
    )
    data = _fetch_with_cache(url, f"weather_{circuit_id}_{race_date}", ttl=86400 * 365)
    daily = data.get("daily", {})
    try:
        temp = float(daily["temperature_2m_max"][0])
        precip = float(daily["precipitation_sum"][0])
        return {"air_temp_c": temp, "precip_mm": precip}
    except (KeyError, IndexError, TypeError, ValueError):
        return {"air_temp_c": 22.0, "precip_mm": 0.0}


def build_weather_lookup(
    races_by_year: dict[int, list],
    progress_cb=None,
) -> dict[tuple, dict]:
    """For training: returns {(year, round): weather} for all known races.

    Fetches in parallel via a thread pool — each call is mostly network I/O,
    and Open-Meteo's free tier handles ~10 RPS comfortably.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs: list[tuple[int, int, str, str]] = []
    for year, races in races_by_year.items():
        for race in races:
            try:
                rnd = int(race.get("round", 0))
            except (TypeError, ValueError):
                continue
            cid = race.get("Circuit", {}).get("circuitId", "")
            date = race.get("date", "")
            jobs.append((year, rnd, cid, date))

    lookup: dict[tuple, dict] = {}
    total = len(jobs)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(get_weather, cid, date): (year, rnd)
            for year, rnd, cid, date in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            year, rnd = futures[fut]
            try:
                lookup[(year, rnd)] = fut.result()
            except Exception:
                lookup[(year, rnd)] = {"air_temp_c": 22.0, "precip_mm": 0.0}
            if progress_cb:
                progress_cb(i, total)
    return lookup
