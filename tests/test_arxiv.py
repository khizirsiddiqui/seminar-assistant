from types import SimpleNamespace

import httpx
import pytest

from app import arxiv


class _OkResponse:
    status_code = 200
    headers = {}


def test_get_with_retry_retries_transport_errors(monkeypatch):
    calls = {"n": 0}
    failures = 2

    class FlakyClient:
        def get(self, url, **kwargs):
            calls["n"] += 1
            if calls["n"] <= failures:
                raise httpx.ReadTimeout("timed out", request=httpx.Request("GET", url))
            return _OkResponse()

    monkeypatch.setattr(arxiv.time, "sleep", lambda _seconds: None)
    response = arxiv._get_with_retry(FlakyClient(), "https://export.arxiv.org/api/query")

    assert calls["n"] == failures + 1
    assert response.status_code == 200


def test_get_with_retry_retries_5xx(monkeypatch):
    statuses = [502, 503, 200]

    class FlakyClient:
        def get(self, url, **kwargs):
            return SimpleNamespace(status_code=statuses.pop(0), headers={})

    monkeypatch.setattr(arxiv.time, "sleep", lambda _seconds: None)
    response = arxiv._get_with_retry(FlakyClient(), "https://export.arxiv.org/api/query")

    assert response.status_code == 200


def test_find_paper_preserves_real_error(monkeypatch):
    class BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, **kwargs):
            raise httpx.ReadTimeout("timed out", request=httpx.Request("GET", url))

    monkeypatch.setattr(arxiv.httpx, "Client", BoomClient)
    monkeypatch.setattr(arxiv.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        arxiv, "get_settings",
        lambda: SimpleNamespace(arxiv_timeout_seconds=5),
    )

    with pytest.raises(RuntimeError, match="Could not reach the arXiv API") as excinfo:
        arxiv.find_paper("Any Query", title="Some Paper Title")

    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)