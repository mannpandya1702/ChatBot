"""T-4.7 verification: websearch.search against a mocked SearXNG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from jarvis.config import JarvisConfig, load_config
from jarvis.tools import websearch as websearch_module
from jarvis.tools.registry import registry
from jarvis.tools.websearch import SearchInput, web_search

SEARXNG_URL = "http://localhost:8080"


@pytest.fixture
def enabled(tmp_path: Path) -> JarvisConfig:
    return load_config(
        tmp_path / "absent.yaml",
        tools={"enable_websearch": True, "searxng_url": SEARXNG_URL},
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
    config: JarvisConfig,
) -> list[httpx.Request]:
    """Point the tool at a mock transport and at ``config``."""
    from jarvis import config as config_module

    seen: list[httpx.Request] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(config_module, "_active", config)
    monkeypatch.setattr(
        websearch_module,
        "_client",
        lambda: httpx.Client(transport=httpx.MockTransport(transport_handler)),
    )
    return seen


def _results_response(count: int = 5) -> Any:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": f"Result {index}",
                        "url": f"https://example.test/{index}",
                        "content": f"Snippet number {index}.",
                        "engine": "duckduckgo",
                    }
                    for index in range(count)
                ]
            },
        )

    return handler


class TestDisabled:
    def test_off_by_default(self, cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch) -> None:
        """§9: web search ships off."""
        from jarvis import config as config_module

        monkeypatch.setattr(config_module, "_active", cfg)
        result = web_search(SearchInput(query="anything"))
        assert result.available is False
        assert "turned off" in (result.reason or "")

    def test_hidden_from_the_model_when_disabled(self, cfg: JarvisConfig) -> None:
        names = [spec.name for spec in registry.select(config=cfg)]
        assert "websearch.search" not in names

    def test_offered_when_enabled(self, enabled: JarvisConfig) -> None:
        names = [spec.name for spec in registry.select(config=enabled)]
        assert "websearch.search" in names

    def test_no_request_is_made_when_disabled(
        self, cfg: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """§0.1: a default install makes no outbound requests at all."""
        seen = _install(monkeypatch, _results_response(), cfg)
        web_search(SearchInput(query="anything"))
        assert seen == []


class TestSearch:
    def test_returns_three_results(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T-4.7 asks for three results with snippets."""
        _install(monkeypatch, _results_response(10), enabled)
        result = web_search(SearchInput(query="weather"))
        assert result.available is True
        assert len(result.results) == 3

    def test_results_carry_snippets(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, _results_response(), enabled)
        first = web_search(SearchInput(query="weather")).results[0]
        assert first.title
        assert first.url.startswith("https://")
        assert first.snippet

    def test_query_reaches_the_instance(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _install(monkeypatch, _results_response(), enabled)
        web_search(SearchInput(query="tallest mountain"))
        assert seen[0].url.params["q"] == "tallest mountain"
        assert seen[0].url.params["format"] == "json"

    def test_category_is_forwarded(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _install(monkeypatch, _results_response(), enabled)
        web_search(SearchInput(query="x", category="news"))
        assert seen[0].url.params["categories"] == "news"

    def test_safesearch_is_on(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = _install(monkeypatch, _results_response(), enabled)
        web_search(SearchInput(query="x"))
        assert seen[0].url.params["safesearch"] == "1"

    def test_entries_missing_a_url_are_skipped(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "no url", "content": "x"},
                        {"url": "https://ok.test", "title": "fine", "content": "y"},
                    ]
                },
            )

        _install(monkeypatch, handler, enabled)
        results = web_search(SearchInput(query="x")).results
        assert [r.title for r in results] == ["fine"]

    def test_snippets_are_trimmed(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snippets get read aloud, so they cannot be a wall of text."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"title": "t", "url": "https://x.test", "content": "y" * 5000}
                    ]
                },
            )

        _install(monkeypatch, handler, enabled)
        assert len(web_search(SearchInput(query="x")).results[0].snippet) <= 300

    def test_empty_results(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": []})

        _install(monkeypatch, handler, enabled)
        result = web_search(SearchInput(query="asdfghjkl"))
        assert result.available is True
        assert result.results == []


class TestFailureModes:
    def test_connection_refused_is_reported_not_raised(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        _install(monkeypatch, handler, enabled)
        result = web_search(SearchInput(query="x"))
        assert result.available is False
        assert "not reachable" in (result.reason or "")

    def test_timeout_is_reported(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        _install(monkeypatch, handler, enabled)
        assert "timed out" in (web_search(SearchInput(query="x")).reason or "")

    def test_json_format_disabled_explains_the_fix(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """403 is what SearXNG returns until the operator enables the json format."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="Forbidden")

        _install(monkeypatch, handler, enabled)
        reason = web_search(SearchInput(query="x")).reason or ""
        assert "settings.yml" in reason

    def test_server_error_is_reported(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        _install(monkeypatch, handler, enabled)
        assert "500" in (web_search(SearchInput(query="x")).reason or "")

    def test_non_json_body_becomes_a_tool_error(
        self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from jarvis.util.errors import ToolExecutionError

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        _install(monkeypatch, handler, enabled)
        with pytest.raises(ToolExecutionError):
            web_search(SearchInput(query="x"))

    def test_missing_url_when_enabled_is_caught_by_config(self, tmp_path: Path) -> None:
        """The config refuses to enable search without an address."""
        from jarvis.util.errors import ConfigError

        with pytest.raises(ConfigError, match="searxng_url"):
            load_config(tmp_path / "absent.yaml", tools={"enable_websearch": True})


class TestRegistration:
    def test_read_only(self) -> None:
        spec = registry.get("websearch.search")
        assert spec is not None
        assert spec.read_only is True

    def test_description_scopes_it_away_from_system_questions(self) -> None:
        spec = registry.get("websearch.search")
        assert spec is not None
        assert "not use it for anything about this computer" in spec.description.lower()

    def test_dispatch(self, enabled: JarvisConfig, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, _results_response(), enabled)
        result = registry.dispatch("websearch.search", {"query": "x"})
        assert result.ok is True
        json.dumps(result.data)
