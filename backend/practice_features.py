"""Practice-session feature extraction using the FastF1 library.

For each Grand Prix, this fetches FP1/FP2/FP3 + Q lap-by-lap data from the
official F1 timing API (free, cached locally) and computes per-driver features
the race-day model otherwise can't see:

  fp_best_lap_norm       : fastest single lap across practice, normalized to pole
  fp_longrun_pace_norm   : median lap time on long runs (≥5 laps clean), normalized
  fp_teammate_quali_gap  : driver's quali time minus their teammate's (skill signal)
  fp_session_laps        : total practice laps completed (setup confidence proxy)
  fp_consistency         : std dev of long-run lap times (race-pace stability)

A driver's 3-letter code (VER, HAM) is mapped to Ergast driver_id via the
session results metadata FastF1 ships with each session.
"""
from __future__ import annotations

import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FILE = Path("data/cache/practice_features.json")
FASTF1_CACHE = Path("data/cache/fastf1")
FASTF1_CACHE.mkdir(parents=True, exist_ok=True)

# Mapping from FastF1's three-letter driver code to Ergast driverId, populated
# lazily from the session metadata. Persisted alongside the features cache.
_code_to_id: dict[str, str] = {}


def _init_fastf1() -> None:
    """Idempotent FastF1 setup; lazy-imported so unit tests don't pay the cost."""
    import fastf1
    fastf1.Cache.enable_cache(str(FASTF1_CACHE))
    # FastF1 is chatty by default. Demote to WARNING so logs stay readable.
    for name in ("fastf1", "fastf1.core", "fastf1._api", "fastf1.req"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"features": {}, "code_to_id": {}}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache))


def _seconds(td) -> float | None:
    """Convert a pandas Timedelta to seconds. Returns None for NaT."""
    if td is None:
        return None
    try:
        s = td.total_seconds()
        if s != s or s <= 0:  # NaN / sentinel
            return None
        return s
    except (AttributeError, TypeError):
        return None


def _extract_session_features(year: int, round_num: int) -> dict:
    """Heavy lifter: pulls FP1/FP2/FP3 + Q for one race and computes per-driver
    summary stats. Returns {} on any unrecoverable failure (early seasons may
    not have FastF1 data; the model gracefully falls back to neutral defaults).
    """
    import fastf1
    import pandas as pd

    _init_fastf1()

    # Aggregate across all practice sessions for richer long-run sample
    practice_laps: list = []
    sessions_loaded = 0
    code_to_team: dict[str, str] = {}
    for session_id in ("FP1", "FP2", "FP3"):
        try:
            s = fastf1.get_session(year, round_num, session_id)
            s.load(laps=True, telemetry=False, weather=False, messages=False)
            if s.laps is None or len(s.laps) == 0:
                continue
            laps = s.laps.copy()
            laps["_session"] = session_id
            practice_laps.append(laps)
            sessions_loaded += 1
            # Stash team mapping while we're here
            if s.results is not None:
                for _, row in s.results.iterrows():
                    code = str(row.get("Abbreviation", "")).upper()
                    team = str(row.get("TeamName", ""))
                    if code and team:
                        code_to_team[code] = team
        except Exception as e:
            logger.debug("FP load failed %s %s %s: %s", year, round_num, session_id, e)
            continue

    # Qualifying — gives the cleanest single-lap pace + driverId mapping
    quali_best: dict[str, float] = {}
    quali_results: dict[str, dict] = {}
    try:
        q = fastf1.get_session(year, round_num, "Q")
        q.load(laps=True, telemetry=False, weather=False, messages=False)
        for _, row in q.results.iterrows():
            code = str(row.get("Abbreviation", "")).upper()
            team = str(row.get("TeamName", ""))
            if code and team:
                code_to_team.setdefault(code, team)
            quali_results[code] = {
                "team": team,
                "Q1": _seconds(row.get("Q1")),
                "Q2": _seconds(row.get("Q2")),
                "Q3": _seconds(row.get("Q3")),
            }
            best = min(
                (v for v in (quali_results[code]["Q3"], quali_results[code]["Q2"],
                              quali_results[code]["Q1"]) if v),
                default=None,
            )
            if best:
                quali_best[code] = best
    except Exception as e:
        logger.debug("Quali load failed %s %s: %s", year, round_num, e)

    if not practice_laps and not quali_best:
        return {}

    # Build per-driver features
    features: dict[str, dict[str, float]] = {}
    drivers = set(quali_best.keys())
    if practice_laps:
        all_p = pd.concat(practice_laps, ignore_index=True)
        drivers.update(str(d).upper() for d in all_p["Driver"].dropna().unique())

    pole_time = min(quali_best.values()) if quali_best else None

    for code in drivers:
        feat: dict[str, float] = {}
        # Quali-derived: gap to pole (already in features but we recompute here
        # using FastF1's higher-precision timing)
        if pole_time and code in quali_best:
            feat["fp_quali_gap_norm"] = quali_best[code] - pole_time

        if practice_laps:
            d_laps = all_p[all_p["Driver"].astype(str).str.upper() == code]
            # Drop pit-in / pit-out laps; they don't reflect real pace
            clean = d_laps[
                d_laps["PitOutTime"].isna() & d_laps["PitInTime"].isna()
            ].copy()
            clean["_secs"] = clean["LapTime"].apply(_seconds)
            clean = clean.dropna(subset=["_secs"])
            if len(clean):
                # Single best lap — "qualifying simulation"
                feat["fp_best_lap"] = float(clean["_secs"].min())
                feat["fp_session_laps"] = float(len(clean))
                # Long-run pace: drop top/bottom 10% then take median of laps
                # within 107% of best (excludes outliers)
                threshold = clean["_secs"].min() * 1.07
                race_pace = clean[clean["_secs"] <= threshold]["_secs"]
                if len(race_pace) >= 3:
                    feat["fp_longrun_pace"] = float(race_pace.median())
                    feat["fp_consistency"] = float(race_pace.std())

        if feat:
            features[code] = feat

    # Normalize fp_best_lap and fp_longrun_pace as gaps to the fastest driver,
    # in seconds. This makes the feature comparable across circuits.
    if features:
        bests = [f["fp_best_lap"] for f in features.values() if "fp_best_lap" in f]
        if bests:
            min_best = min(bests)
            for f in features.values():
                if "fp_best_lap" in f:
                    f["fp_best_lap_norm"] = f.pop("fp_best_lap") - min_best
        longs = [f["fp_longrun_pace"] for f in features.values() if "fp_longrun_pace" in f]
        if longs:
            min_long = min(longs)
            for f in features.values():
                if "fp_longrun_pace" in f:
                    f["fp_longrun_pace_norm"] = f.pop("fp_longrun_pace") - min_long

    # Teammate quali gap
    team_codes: dict[str, list[str]] = {}
    for code, team in code_to_team.items():
        team_codes.setdefault(team, []).append(code)
    for _team, codes in team_codes.items():
        if len(codes) != 2:
            continue
        a, b = codes
        ta = quali_best.get(a)
        tb = quali_best.get(b)
        if ta and tb:
            features.setdefault(a, {})["fp_teammate_quali_gap"] = ta - tb
            features.setdefault(b, {})["fp_teammate_quali_gap"] = tb - ta

    return {"features": features, "code_to_team": code_to_team}


def fetch_practice_features(
    year: int,
    round_num: int,
    cache: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Returns {driver_code: {feature_name: value}}. Cached on disk forever.
    Driver code is the 3-letter FastF1 abbreviation (VER, LEC, HAM).
    """
    cache = cache if cache is not None else _load_cache()
    key = f"{year}_{round_num}"
    if key in cache["features"]:
        return cache["features"][key]

    try:
        out = _extract_session_features(year, round_num)
    except Exception as e:
        logger.warning("Practice features failed for %s R%s: %s", year, round_num, e)
        out = {}

    cache["features"][key] = out.get("features", {})
    # Merge code→team into cache (used to map to driver_id later)
    cache.setdefault("code_to_team", {}).update(out.get("code_to_team", {}))
    _save_cache(cache)
    return cache["features"][key]


def build_practice_lookup(
    races_by_year: dict[int, list],
    progress_cb=None,
) -> dict[tuple, dict[str, dict[str, float]]]:
    """For training: returns {(year, round): {driver_code: features}}.
    FastF1's official data starts in 2018, which happens to match our training range.
    """
    cache = _load_cache()
    jobs: list[tuple[int, int]] = []
    for year, races in races_by_year.items():
        if year < 2018:
            continue
        for race in races:
            try:
                rnd = int(race.get("round", 0))
            except (TypeError, ValueError):
                continue
            jobs.append((year, rnd))

    out: dict[tuple, dict] = {}
    total = len(jobs)
    # FastF1 is rate-limited; 3 workers is plenty
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fetch_practice_features, y, r, cache): (y, r) for y, r in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                out[key] = fut.result()
            except Exception:
                out[key] = {}
            if progress_cb:
                progress_cb(i, total)
    return out


# ── Driver code → Ergast driver_id mapping ─────────────────────────────


def build_code_to_id_mapping(races_by_year: dict[int, list]) -> dict[str, str]:
    """Walk every race in the training set and derive a {three-letter code:
    Ergast driver_id} mapping from Ergast's own data. The Ergast Driver object
    has a "code" field that matches FastF1's Abbreviation."""
    mapping: dict[str, str] = {}
    for races in races_by_year.values():
        for race in races:
            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                code = driver.get("code", "")
                did = driver.get("driverId", "")
                if code and did:
                    mapping[code.upper()] = did
    return mapping
