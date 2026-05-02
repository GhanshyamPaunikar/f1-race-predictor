import httpx
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "https://api.jolpi.ca/ergast/f1"


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
            resp = client.get(url, headers={"User-Agent": "F1RacePredictor/1.0"})
            resp.raise_for_status()
            data = resp.json()
        cf.write_text(json.dumps(data))
        return data
    except Exception as e:
        logger.warning(f"API fetch failed for {url}: {e}")
        if cf.exists():
            return json.loads(cf.read_text())
        return {}


def get_race_results(year: int) -> list:
    results = []
    offset = 0
    limit = 100
    while True:
        url = f"{BASE_URL}/{year}/results.json?limit={limit}&offset={offset}"
        data = _fetch_with_cache(url, f"results_{year}_{offset}", ttl=86400)
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            break
        results.extend(races)
        total = int(data.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break
    return results


def get_qualifying(year: int) -> list:
    results = []
    offset = 0
    limit = 100
    while True:
        url = f"{BASE_URL}/{year}/qualifying.json?limit={limit}&offset={offset}"
        data = _fetch_with_cache(url, f"quali_{year}_{offset}", ttl=86400)
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if not races:
            break
        results.extend(races)
        total = int(data.get("MRData", {}).get("total", 0))
        offset += limit
        if offset >= total:
            break
    return results


def get_schedule(year: int) -> list:
    url = f"{BASE_URL}/{year}.json"
    data = _fetch_with_cache(url, f"schedule_{year}", ttl=86400)
    return data.get("MRData", {}).get("RaceTable", {}).get("Races", [])


def get_driver_standings(year: int, round_num: int = None) -> list:
    if round_num:
        url = f"{BASE_URL}/{year}/{round_num}/driverStandings.json"
        key = f"driver_standings_{year}_{round_num}"
    else:
        url = f"{BASE_URL}/{year}/driverStandings.json"
        key = f"driver_standings_{year}"
    data = _fetch_with_cache(url, key, ttl=3600)
    lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("DriverStandings", []) if lists else []


def get_constructor_standings(year: int, round_num: int = None) -> list:
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
