"""Shared HTTP session with retries for the API loaders."""

from __future__ import annotations

from typing import Any, Protocol

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "gb-price-sandbox/0.1"


class JsonSession(Protocol):
    """The subset of ``requests.Session`` the loaders use (lets tests pass a fake)."""

    def get(self, url: str, *, params: dict[str, Any] | None = ..., timeout: float = ...) -> Any:
        """Issue a GET request."""
        ...


def make_session() -> requests.Session:
    """Create a session that retries transient failures with exponential backoff."""
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get_json(session: JsonSession, url: str, params: dict[str, Any] | None, timeout: float) -> Any:
    """GET ``url`` and return the decoded JSON body, raising on HTTP errors."""
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()
