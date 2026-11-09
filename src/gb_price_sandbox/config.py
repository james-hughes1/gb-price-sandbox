"""Runtime configuration, read from environment variables.

All settings use the ``GBPS_`` prefix. No secrets live in the repo; see ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings.

    Attributes:
        data_dir: Root directory for the local parquet cache.
        http_timeout: Timeout in seconds for API requests.
        elexon_base_url: Base URL of the Elexon Insights API.
    """

    data_dir: Path = Path("data")
    http_timeout: float = 30.0
    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        """Build settings from environment variables, falling back to defaults.

        Args:
            env: Mapping to read from. Defaults to ``os.environ``.

        Returns:
            A populated ``Settings`` instance.
        """
        env = dict(os.environ) if env is None else env
        defaults = cls()
        return cls(
            data_dir=Path(env.get("GBPS_DATA_DIR", defaults.data_dir)),
            http_timeout=float(env.get("GBPS_HTTP_TIMEOUT", defaults.http_timeout)),
            elexon_base_url=env.get("GBPS_ELEXON_BASE_URL", defaults.elexon_base_url),
        )


def get_settings() -> Settings:
    """Return settings for the current process environment."""
    return Settings.from_env()
