"""Web search via a self-hosted SearXNG instance.

T-4.7, and one of the few places §0.1 permits network access: SearXNG is
explicitly user-enabled, self-hosted, and carries no API key or account. The
tool stays disabled until ``tools.searxng_url`` is configured, so a default
install makes no outbound requests at all.

Read-only. It searches and summarises; it never opens a page or downloads a file.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import Field

from jarvis.config import get_config
from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import ToolExecutionError

__all__ = ["SearchInput", "SearchOutput", "SearchResult", "web_search"]

_log = logging.getLogger(__name__)

#: T-4.7 asks for three results with snippets.
_MAX_RESULTS = 3
_TIMEOUT_S = 15.0
#: Snippets get read aloud, so they are trimmed to something speakable.
_MAX_SNIPPET_CHARS = 300


class SearchResult(ToolOutput):
    """One search hit."""

    title: str = Field(description="Page title.")
    url: str = Field(description="Page address.")
    snippet: str = Field(description="Short extract, trimmed for speech.")
    engine: str | None = Field(default=None, description="Which upstream engine supplied it.")


class SearchInput(ToolInput):
    """Arguments for the web search tool."""

    query: str = Field(
        min_length=1,
        max_length=300,
        description="What to search for, phrased as a search query rather than a question.",
    )
    category: str | None = Field(
        default=None,
        description="Optional SearXNG category, for example news, science, or it.",
    )


class SearchOutput(ToolOutput):
    """Search results, or an explanation of why the search did not run."""

    available: bool = Field(
        description="False when web search is disabled or the instance is unreachable."
    )
    reason: str | None = Field(default=None, description="Why the search did not run.")
    query: str = Field(description="What was searched for.")
    results: list[SearchResult] = Field(
        default_factory=list, description="Up to 3 results with snippets."
    )


#: How many same-host redirects to follow. A SearXNG instance behind a proxy
#: may legitimately bounce http to https on the same host; more hops than this
#: is a loop or a misconfiguration.
_MAX_REDIRECTS = 3


def _client() -> httpx.Client:
    """HTTP client for the configured SearXNG instance.

    Redirects are followed manually rather than by httpx, so that a hop to a
    different host can be refused. §0.1 permits network access to the SearXNG
    instance the user configured and to nothing else, and httpx following a
    redirect off-host would send the user's query to an address they never
    named.
    """
    return httpx.Client(
        timeout=httpx.Timeout(_TIMEOUT_S, connect=5.0),
        follow_redirects=False,
        headers={"User-Agent": "jarvis-local-assistant"},
    )


def _same_origin(first: httpx.URL, second: httpx.URL) -> bool:
    """Whether two URLs address the same host and port.

    The scheme is allowed to change so an http to https upgrade still works;
    the host and the effective port are not.
    """
    return first.host == second.host and first.port == second.port


def _redirect_refusal(response: httpx.Response) -> str:
    """Say why an unfollowed redirect stopped the search, accurately.

    Off-host and redirect-loop are different faults and read differently to
    someone trying to fix their instance, so they are not collapsed into one
    message.
    """
    target = response.headers.get("location")
    if target and not _same_origin(response.url.join(target), response.url):
        return "the SearXNG instance redirected the search to a different address"
    return "the SearXNG instance kept redirecting the search"


def _get_following_same_host_redirects(
    client: httpx.Client, url: str, params: dict[str, str | int]
) -> httpx.Response:
    """GET ``url``, following redirects only while they stay on the same host.

    Returns:
        The final response. A redirect pointing at another host is returned
        as-is, unfollowed, so the caller reports it rather than chasing it.
    """
    response = client.get(url, params=params)
    for _ in range(_MAX_REDIRECTS):
        if not response.is_redirect:
            return response
        target = response.headers.get("location")
        if not target:
            return response
        destination = response.url.join(target)
        if not _same_origin(destination, response.url):
            _log.warning(
                "refusing a search redirect to another host",
                extra={"context": {"from": response.url.host, "to": destination.host}},
            )
            return response
        response = client.get(destination)
    return response


@tool(
    name="websearch.search",
    description=(
        "Search the web through the user's own SearXNG instance and return three results "
        "with short extracts. Use this only for questions about the outside world that no "
        "system tool can answer, such as current events or factual lookups. Do not use it "
        "for anything about this computer. Disabled unless the user has configured a SearXNG "
        "address. Read-only, it never opens pages or downloads files."
    ),
    category=ToolCategory.WEB,
    read_only=True,
    is_enabled=lambda config: config.tools.enable_websearch and bool(config.tools.searxng_url),
)
def web_search(params: SearchInput) -> SearchOutput:
    """Query SearXNG.

    Args:
        params: The query and an optional category.

    Returns:
        Up to three results, or ``available=False`` with a reason.

    Raises:
        ToolExecutionError: The response could not be processed at all.
    """
    config = get_config()

    if not config.tools.enable_websearch:
        return SearchOutput(
            available=False,
            reason="web search is turned off in the configuration",
            query=params.query,
        )
    base_url = config.tools.searxng_url
    if not base_url:
        return SearchOutput(
            available=False,
            reason="no SearXNG address is configured",
            query=params.query,
        )

    request: dict[str, str | int] = {
        "q": params.query,
        "format": "json",
        "safesearch": 1,
    }
    if params.category:
        request["categories"] = params.category

    try:
        with _client() as client:
            response = _get_following_same_host_redirects(
                client, f"{base_url.rstrip('/')}/search", request
            )
        if response.is_redirect:
            return SearchOutput(
                available=False,
                reason=_redirect_refusal(response),
                query=params.query,
            )
    except httpx.ConnectError:
        return SearchOutput(
            available=False,
            reason="the SearXNG instance is not reachable",
            query=params.query,
        )
    except httpx.TimeoutException:
        return SearchOutput(
            available=False, reason="the search timed out", query=params.query
        )
    except httpx.HTTPError as exc:
        _log.warning("websearch request failed: %s", exc)
        return SearchOutput(
            available=False, reason="the search request failed", query=params.query
        )

    if response.status_code == 403:
        # SearXNG rejects the JSON format unless the operator enables it.
        return SearchOutput(
            available=False,
            reason="the SearXNG instance refused a JSON request; enable the json format in "
            "its settings.yml",
            query=params.query,
        )
    if response.status_code >= 400:
        return SearchOutput(
            available=False,
            reason=f"the SearXNG instance returned status {response.status_code}",
            query=params.query,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise ToolExecutionError(
            "websearch.search",
            f"SearXNG returned a non-JSON response: {exc}",
            speakable="The search came back in a form I could not read.",
        ) from exc

    results: list[SearchResult] = []
    for entry in payload.get("results", [])[: _MAX_RESULTS * 3]:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        title = str(entry.get("title") or "").strip()
        if not url or not title:
            continue
        snippet = str(entry.get("content") or "").strip()
        results.append(
            SearchResult(
                title=title[:200],
                url=url,
                snippet=snippet[:_MAX_SNIPPET_CHARS],
                engine=str(entry.get("engine")) if entry.get("engine") else None,
            )
        )
        if len(results) >= _MAX_RESULTS:
            break

    return SearchOutput(available=True, query=params.query, results=results)
