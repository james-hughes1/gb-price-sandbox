"""Test doubles shared across loader tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class FakeResponse:
    def __init__(self, payload: Any):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Records every GET and answers it with ``handler(url, params)``."""

    def __init__(self, handler: Callable[[str, dict | None], Any]):
        self.handler = handler
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, *, params: dict | None = None, timeout: float = 0) -> FakeResponse:
        self.calls.append((url, params))
        return FakeResponse(self.handler(url, params))
