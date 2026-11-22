"""Pretrained forecasts for the Battery optimiser.

The Battery page never trains anything. It reads walk-forward forecasts saved here by
:func:`pretrain`: one naive model and one learning model, over every delivery day from the
start of the holdout to the most recent month boundary. Run it from the command line:

    uv run python -m gb_price_sandbox.pretrain

It is idempotent (each backtest is cached), takes well under a minute from scratch, and the
Docker image runs it on start-up.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gb_price_sandbox.backtest import cached_walk_forward
from gb_price_sandbox.config import Settings
from gb_price_sandbox.models import MODEL_SPECS_BY_NAME, MODELS

#: The two models the Battery page compares: a naive benchmark and a learning model.
PRETRAINED_MODELS: tuple[str, ...] = ("Seasonal naive", "LEAR")


def pretrained_dir(settings: Settings) -> Path:
    """Folder holding the pretrained forecasts and their manifest."""
    return settings.data_dir / "processed" / "pretrained"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower())


@dataclass(frozen=True)
class Pretrained:
    """Saved forecasts and when they cover."""

    results: dict[str, pd.DataFrame]
    test_start: dt.date
    test_end: dt.date
    trained_at: str


def pretrain(
    data: pd.DataFrame,
    test_start: dt.date,
    test_end: dt.date,
    settings: Settings,
    *,
    models: tuple[str, ...] = PRETRAINED_MODELS,
    progress: Callable[[str], None] | None = None,
) -> Pretrained:
    """Backtest ``models`` over ``[test_start, test_end)`` and save the forecasts.

    Uses each model's default configuration, so the results match the Forecast sandbox's
    defaults (and share its cache).
    """
    folder = pretrained_dir(settings)
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    results = {}
    for name in models:
        if progress is not None:
            progress(f"Backtesting {name}…")
        spec = MODEL_SPECS_BY_NAME[name]
        res = cached_walk_forward(
            data,
            name,
            MODELS[name],
            test_start,
            test_end,
            settings,
            retrain_every_days=spec.retrain_every_days,
            max_train_days=spec.max_train_days,
        )
        path = folder / f"{_slug(name)}.parquet"
        res.to_parquet(path)
        files[name] = path.name
        results[name] = res
    trained_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    manifest = {
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "trained_at": trained_at,
        "models": files,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return Pretrained(results, test_start, test_end, trained_at)


def load_pretrained(settings: Settings) -> Pretrained | None:
    """The saved forecasts, or ``None`` if :func:`pretrain` hasn't been run."""
    folder = pretrained_dir(settings)
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    results = {}
    for name, file in manifest["models"].items():
        if not (folder / file).exists():
            return None
        res = pd.read_parquet(folder / file)
        res["delivery_day"] = pd.to_datetime(res["delivery_day"]).dt.date
        results[name] = res
    return Pretrained(
        results,
        dt.date.fromisoformat(manifest["test_start"]),
        dt.date.fromisoformat(manifest["test_end"]),
        manifest["trained_at"],
    )
