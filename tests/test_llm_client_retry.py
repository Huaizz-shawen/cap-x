from __future__ import annotations

import requests

from capx.llm.client import ModelQueryArgs, query_model


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}", response=self)

    def json(self) -> dict:
        return self._body


def test_query_model_retries_on_timeout(monkeypatch):
    calls = {"count": 0}
    sleeps: list[float] = []

    def _fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ReadTimeout("read timed out")
        return _FakeResponse(
            200,
            {"choices": [{"message": {"content": "ok", "reasoning": None}}]},
        )

    monkeypatch.setattr("capx.llm.client.requests.post", _fake_post)
    monkeypatch.setattr("capx.llm.client.time.sleep", sleeps.append)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda a, b: 0.0)
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CAPX_MODEL_RETRY_INITIAL_S", "1")
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_SLEEP_S", "2")
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_WALLTIME_S", "30")

    out = query_model(
        ModelQueryArgs(model="gemini-3-pro-preview", server_url="http://example.test/v1/chat/completions"),
        [{"role": "user", "content": "hello"}],
    )

    assert out["content"] == "ok"
    assert calls["count"] == 2
    assert sleeps == [1.0]


def test_query_model_retries_on_503(monkeypatch):
    responses = [
        _FakeResponse(503, {"error": "busy"}),
        _FakeResponse(200, {"choices": [{"message": {"content": "done", "reasoning": None}}]}),
    ]
    sleeps: list[float] = []

    def _fake_post(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", _fake_post)
    monkeypatch.setattr("capx.llm.client.time.sleep", sleeps.append)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda a, b: 0.0)
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CAPX_MODEL_RETRY_INITIAL_S", "2")
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_SLEEP_S", "4")
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_WALLTIME_S", "30")

    out = query_model(
        ModelQueryArgs(model="gemini-3-pro-preview", server_url="http://example.test/v1/chat/completions"),
        [{"role": "user", "content": "hello"}],
    )

    assert out["content"] == "done"
    assert sleeps == [2.0]


def test_query_model_retries_on_525(monkeypatch):
    responses = [
        _FakeResponse(525, {"error": "ssl handshake failed"}),
        _FakeResponse(200, {"choices": [{"message": {"content": "recovered", "reasoning": None}}]}),
    ]
    sleeps: list[float] = []

    def _fake_post(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("capx.llm.client.requests.post", _fake_post)
    monkeypatch.setattr("capx.llm.client.time.sleep", sleeps.append)
    monkeypatch.setattr("capx.llm.client.random.uniform", lambda a, b: 0.0)
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("CAPX_MODEL_RETRY_INITIAL_S", "2")
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_SLEEP_S", "4")
    monkeypatch.setenv("CAPX_MODEL_RETRY_MAX_WALLTIME_S", "30")

    out = query_model(
        ModelQueryArgs(model="gemini-3-pro-preview", server_url="http://example.test/v1/chat/completions"),
        [{"role": "user", "content": "hello"}],
    )

    assert out["content"] == "recovered"
    assert sleeps == [2.0]
