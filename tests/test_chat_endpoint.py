"""HTTP-level tests for the streaming /chat endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_ready_after_startup():
    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True


def test_chat_streams_sse_events():
    with TestClient(app) as client:
        resp = client.post("/chat", json={"message": "When is the launch date?"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert '"type": "token"' in body
        assert '"type": "metadata"' in body
        assert '"route": "static"' in body


def test_chat_rejects_empty_message():
    with TestClient(app) as client:
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422  # Pydantic validation (min_length=1)
