"""Feature sets the Forecast sandbox can hand to the models.

The **core** set is what the models were built around and is always on. Each optional group
adds columns from :mod:`gb_price_sandbox.features.explore`. Only groups whose every column is
known at the D-1 issue time are offered: the Open-Meteo weather series (a same-day forecast)
and PV_Live outturn are deliberately absent. ``tests/test_feature_sets.py`` checks this.
"""

from __future__ import annotations

from dataclasses import dataclass

from gb_price_sandbox.features.dataset import FEATURE_COLUMNS


@dataclass(frozen=True)
class FeatureGroup:
    """A group of columns that can be switched on together.

    Attributes:
        key: Stable identifier.
        label: Name in the selector.
        columns: Columns added.
        price_like: The subset measured in £/MWh, which linear models put on the asinh
            scale (the rest are standardised).
        description: One line for the selector's help text.
    """

    key: str
    label: str
    columns: tuple[str, ...]
    price_like: tuple[str, ...]
    description: str


CORE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS

FEATURE_GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup(
        "calendar",
        "Holidays and season",
        ("is_holiday", "doy_sin", "doy_cos"),
        (),
        "Bank holiday / Christmas flag and the time of year as a sine-cosine pair.",
    ),
    FeatureGroup(
        "recent_prices",
        "Recent price shape",
        (
            "price_d1_am_mean",
            "price_d1_last",
            "price_d2_spread",
            "price_vol_7d",
            "price_neg_share_7d",
        ),
        ("price_d1_am_mean", "price_d1_last", "price_d2_spread", "price_vol_7d"),
        "D-1 morning prices, how peaky D-2 was, and last week's volatility.",
    ),
    FeatureGroup(
        "residual_shape",
        "Residual-demand shape",
        ("residual_ramp_gw", "residual_day_max_gw", "residual_day_min_gw"),
        (),
        "How fast residual demand ramps, and D's peak and trough (from NESO forecasts).",
    ),
    FeatureGroup(
        "solar",
        "Solar (2 days earlier)",
        ("solar_lag_2d_gw",),
        (),
        "PV_Live solar output in the same half-hour two days earlier.",
    ),
)

FEATURE_GROUPS_BY_KEY: dict[str, FeatureGroup] = {g.key: g for g in FEATURE_GROUPS}


@dataclass(frozen=True)
class FeatureSet:
    """The columns a model may use: the core set plus some optional groups."""

    groups: tuple[str, ...] = ()

    @property
    def extra(self) -> tuple[str, ...]:
        """Columns added on top of the core set, in group order."""
        return tuple(c for k in self.groups for c in FEATURE_GROUPS_BY_KEY[k].columns)

    @property
    def extra_price_like(self) -> tuple[str, ...]:
        """Added columns measured in £/MWh."""
        return tuple(c for k in self.groups for c in FEATURE_GROUPS_BY_KEY[k].price_like)

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column the models may see."""
        return (*CORE_COLUMNS, *self.extra)

    @property
    def label(self) -> str:
        """Short description, e.g. "Core + Holidays and season"."""
        names = [FEATURE_GROUPS_BY_KEY[k].label for k in self.groups]
        return " + ".join(["Core", *names])


CORE = FeatureSet()
