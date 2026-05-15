"""API smoke tests using FastAPI's TestClient with httpx mocked out."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(trained_predictor, monkeypatch):
    """Mount the FastAPI app with a pre-trained predictor and stubbed network."""
    import main
    main.state.predictor = trained_predictor
    main.state.training_status = {"state": "done", "progress": 100, "message": "ok"}

    # Stub all external API calls so tests are hermetic
    fake_schedule = [
        {"round": str(r), "raceName": f"Round {r} GP",
         "date": "2024-04-15",
         "Circuit": {"circuitId": "monza", "circuitName": "Monza",
                     "Location": {"country": "Italy", "locality": "Monza"}}}
        for r in range(1, 6)
    ]
    fake_quali = [
        {"position": str(i + 1),
         "Driver": {"driverId": did, "givenName": did.title(), "familyName": "X",
                    "permanentNumber": "1", "nationality": "X"},
         "Constructor": {"constructorId": tid, "name": tid.title()}}
        for i, (did, tid) in enumerate([
            ("verstappen", "red_bull"), ("hamilton", "mercedes"),
            ("leclerc", "ferrari"), ("norris", "mclaren"),
        ])
    ]
    fake_results = [
        {"position": str(i + 1), "grid": str(i + 1), "points": "25",
         "status": "Finished", "laps": "53",
         "Driver": {"driverId": d["Driver"]["driverId"],
                    "givenName": d["Driver"]["givenName"],
                    "familyName": d["Driver"]["familyName"]},
         "Constructor": d["Constructor"]}
        for i, d in enumerate(fake_quali)
    ]

    monkeypatch.setattr(main.df, "get_schedule", lambda y: fake_schedule)
    monkeypatch.setattr(main.df, "get_round_qualifying", lambda y, r: fake_quali)
    monkeypatch.setattr(main.df, "get_round_result", lambda y, r: fake_results)
    monkeypatch.setattr(main.df, "get_driver_standings", lambda y, r=None: [])
    monkeypatch.setattr(main.df, "get_constructor_standings", lambda y, r=None: [])
    monkeypatch.setattr(main.df, "get_weather", lambda c, d: {"air_temp_c": 24.0, "precip_mm": 0.0})

    with TestClient(main.app) as c:
        yield c


def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["model_trained"] is True
    assert body["metrics"] is not None


def test_predict_endpoint_shape(client):
    r = client.get("/api/predict/2024/3")
    assert r.status_code == 200
    body = r.json()
    assert body["race"]["round"] == 3
    assert body["race"]["weather"]["air_temp_c"] == 24.0
    assert len(body["predictions"]) == 4
    # Win probabilities should be present and sum near 1
    total = sum(p["win_probability"] for p in body["predictions"])
    assert abs(total - 1.0) < 0.05


def test_predict_404_for_missing_round(client):
    r = client.get("/api/predict/2024/99")
    assert r.status_code == 404


def test_explain_endpoint(client):
    r = client.get("/api/explain/2024/3/verstappen")
    assert r.status_code == 200
    body = r.json()
    assert body["driver_id"] == "verstappen"
    assert len(body["contributions"]) > 0


def test_compare_requires_two_drivers(client):
    r = client.get("/api/compare/2024/3?drivers=verstappen")
    assert r.status_code == 400


def test_compare_endpoint(client):
    r = client.get("/api/compare/2024/3?drivers=verstappen,hamilton")
    assert r.status_code == 200
    body = r.json()
    assert len(body["drivers"]) == 2
    for d in body["drivers"]:
        assert "win_probability" in d
        assert "dnf_probability" in d


def test_schedule_endpoint(client):
    r = client.get("/api/schedule/2024")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 5
    assert all("circuit" in i for i in items)


def test_driver_detail_endpoint(client):
    r = client.get("/api/driver/verstappen")
    assert r.status_code == 200
    body = r.json()
    assert body["driver_id"] == "verstappen"
    assert body["races"] >= 0
