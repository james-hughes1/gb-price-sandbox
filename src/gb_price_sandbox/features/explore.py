"""Feature exploration: every feature the models use, plus candidates, with their transforms.

:func:`add_exploration_features` takes the model dataset (see
:mod:`gb_price_sandbox.features.dataset`) and optional extra sources (weather, solar outturn,
bank holidays) and adds one column per *variant*: a raw series and the transformations of it a
model might use. :data:`CATALOGUE` describes each feature for the Features tab: its variants,
how to chart it, whether a model uses it, and when it is known.

Nothing here feeds the backtest. The model inputs are still exactly
:data:`gb_price_sandbox.features.dataset.FEATURE_COLUMNS`. Every derived feature is built only
from data available at the D-1 issue time, except where a variant says otherwise (realised
solar, short-lead weather forecasts), and the tests check this.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from gb_price_sandbox.data.calendar import is_christmas_period
from gb_price_sandbox.timing import LONDON, MID_PUBLICATION_MARGIN

#: Hours between the end of the last D-1 price period used and the issue time, to allow for
#: publication delay. With a 10:00 issue time, D-1 prices up to 09:00 are used.
PUBLICATION_MARGIN_H = int(MID_PUBLICATION_MARGIN / pd.Timedelta(hours=1))
#: Window (days) for the recent-volatility features, ending on D-2.
VOLATILITY_DAYS = 7
#: Lags (days) in the correlation-by-lag chart. One day is not allowed: most of D-1 hasn't
#: happened by the 10:00 issue time.
LAG_PROFILE_DAYS: tuple[int, ...] = tuple(range(2, 15))
#: Lags the models use today.
USED_LAG_DAYS: tuple[int, ...] = (2, 3, 7)
#: Base temperature (°C) for heating degrees: the usual UK convention.
HEATING_BASE_C = 15.5

DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
DAY_TYPES = ("Weekday", "Weekend", "Bank holiday", "Christmas period")
PERIOD_LABELS = tuple(f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30))

Kind = Literal["continuous", "categorical"]
Status = Literal["target", "used", "built", "candidate"]

STATUS_LABELS: dict[str, str] = {
    "target": "Target",
    "used": "Used by a model today",
    "built": "Built but not used",
    "candidate": "Candidate",
}


@dataclass(frozen=True)
class Variant:
    """One column shown for a feature: the raw series or a transformation of it.

    Attributes:
        column: Column in the exploration frame.
        label: Long name for chart titles.
        short: Short name for the correlation matrix.
        unit: Unit for axis titles ("" if unitless).
        kind: ``continuous`` (histogram + relationship to price) or ``categorical`` (price by
            category).
        target: Column to plot it against: ``price`` or, for asinh-scale variants,
            ``price_asinh``.
        in_matrix: Include in the correlation matrix.
        primary: Include in the compact correlation matrix (the rest appear only when
            transformed versions are switched on).
        categories: Display labels for category codes 0..n-1 (categorical only).
        known_at_issue: False for reference series a forecaster could not have had.
        status: Whether a model uses this variant, if it differs from the page's status.
        show: Chart it on the page. False keeps it for the correlation matrix only (e.g.
            asinh versions of every lag, or 0/1 flags already shown as categories).
    """

    column: str
    label: str
    short: str
    unit: str = ""
    kind: Kind = "continuous"
    target: str = "price"
    in_matrix: bool = True
    primary: bool = False
    categories: tuple[str, ...] = ()
    known_at_issue: bool = True
    status: Status | None = None
    show: bool = True


@dataclass(frozen=True)
class DataSource:
    """Where a page's raw data comes from, with a link to its documentation.

    Attributes:
        name: The source as named in the sentence (e.g. "Elexon's Market Index Price").
        link_text: Text of the link.
        url: Documentation or data page.
    """

    name: str
    link_text: str
    url: str


MID = DataSource(
    "Elexon's Market Index Price (MID) dataset",
    "Elexon's MID documentation",
    "https://bmrs.elexon.co.uk/api-documentation/endpoint/balancing/pricing/market-index",
)
NDF = DataSource(
    "NESO's demand forecast (NDF)",
    "Elexon's NDF page",
    "https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/NDF",
)
WINDFOR = DataSource(
    "NESO's wind forecast (WINDFOR)",
    "Elexon's WINDFOR page",
    "https://bmrs.elexon.co.uk/api-documentation/endpoint/datasets/WINDFOR",
)
OPEN_METEO = DataSource(
    "Open-Meteo's Historical Forecast API",
    "Open-Meteo's documentation",
    "https://open-meteo.com/en/docs/historical-forecast-api",
)
PV_LIVE = DataSource(
    "Sheffield Solar's PV_Live",
    "PV_Live",
    "https://www.solar.sheffield.ac.uk/pvlive/",
)
BANK_HOLIDAYS = DataSource(
    "GOV.UK's list of bank holidays (England and Wales)",
    "GOV.UK bank holidays",
    "https://www.gov.uk/bank-holidays",
)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


@dataclass(frozen=True)
class FeatureSpec:
    """A feature (or family of closely related ones) on the Features tab.

    Attributes:
        key: Stable identifier.
        group: Selector heading.
        title: Display name.
        status: Whether a model uses it.
        variants: Raw series first, then transformations.
        summary: One-line description under the selector.
        explanation: Markdown shown under the charts.
        timing: Markdown: when the value is known relative to the D-1 issue time.
        source: Data source.
        models: Models that use the feature today.
        lag_profile: Show the correlation of price with its own lags, day by day.
        timeseries: Show the time-series panel (pointless for lags, which are the price
            series shifted).
        links: Raw data sources, cited with links under the explanation.
        derived_note: Optional sentence on what is computed rather than downloaded.
    """

    key: str
    group: str
    title: str
    status: Status
    variants: tuple[Variant, ...]
    summary: str
    explanation: str
    timing: str
    source: str
    models: tuple[str, ...] = field(default=())
    lag_profile: bool = False
    timeseries: bool = True
    links: tuple[DataSource, ...] = ()
    derived_note: str = ""

    def source_note(self) -> str:
        """Markdown sentence citing the page's raw data sources, with links."""
        names = _join([d.name for d in self.links])
        links = _join([f"[{d.link_text}]({d.url})" for d in self.links])
        note = f"The raw data was sourced from {names}, see {links} for more details."
        return f"{note} {self.derived_note}".strip()

    def variant_status(self, variant: Variant) -> Status:
        """The variant's own status, falling back to the page's."""
        return variant.status or self.status

    def shown(self) -> list[Variant]:
        """Variants charted on the page."""
        return [v for v in self.variants if v.show]


# --- transforms ------------------------------------------------------------------------------


def asinh_scale(values: pd.Series | np.ndarray, median: float, mad: float) -> np.ndarray:
    """LEAR's variance-stabilising transform: ``asinh((x - median) / MAD)``."""
    return np.arcsinh((np.asarray(values, dtype=float) - median) / mad)


def robust_centre(price: pd.Series) -> tuple[float, float]:
    """Median and median absolute deviation of ``price`` (MAD falls back to 1)."""
    p = price.dropna()
    median = float(np.median(p))
    mad = float(np.median(np.abs(p - median))) or 1.0
    return median, mad


def _local_clock_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert(LONDON)
    return (local.hour * 60 + local.minute).to_numpy()


def _shift_daily(daily: pd.Series, days: int, delivery: pd.Series) -> np.ndarray:
    """Map a per-delivery-day series onto rows, taking the value from ``days`` days earlier."""
    shifted = pd.Series(daily.to_numpy(), index=[d + dt.timedelta(days=days) for d in daily.index])
    return delivery.map(shifted).astype(float).to_numpy()


def d1_morning_prices(
    price: pd.Series, delivery: pd.Series, *, hour: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and last price of D-1 periods that have ended (and been published) by the cutoff.

    A period counts if it ends at least :data:`PUBLICATION_MARGIN_H` hours before the D-1
    ``hour``:00 issue time, i.e. with the default issue time, D-1 periods from 00:00 to 08:30.

    Args:
        price: Half-hourly price on a regular UTC grid.
        delivery: Delivery day of each row (same index).
        hour: Issue hour, London time.

    Returns:
        ``(mean, last)`` arrays aligned with ``price``: for each row on day D, the mean and the
        last non-missing price among D-1's eligible periods.
    """
    end_minutes = _local_clock_minutes(price.index) + 30
    eligible = end_minutes <= (hour - PUBLICATION_MARGIN_H) * 60
    p = price[eligible]
    days = delivery[eligible]
    mean = p.groupby(days).mean()
    last = p.dropna().groupby(days[p.notna()]).last()
    return _shift_daily(mean, 1, delivery), _shift_daily(last, 1, delivery)


def trailing_daily_stats(
    values: pd.Series, delivery: pd.Series, window_days: int, lag_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation and share-below-zero of ``values`` over a trailing window of days.

    For a row on day D the window is the ``window_days`` delivery days ending on ``D -
    lag_days``. Days are counted on the calendar, so a missing day shrinks the sample rather
    than stretching the window.

    Returns:
        ``(std, negative_share)`` arrays aligned with ``values``.
    """
    v = values.astype(float)
    frame = pd.DataFrame(
        {"n": v.notna().astype(float), "s": v.fillna(0.0), "ss": v.fillna(0.0) ** 2},
        index=values.index,
    )
    frame["neg"] = (v < 0).astype(float)
    daily = frame.groupby(delivery.to_numpy()).sum()
    all_days = pd.Index(pd.date_range(daily.index.min(), daily.index.max(), freq="D").date)
    rolled = daily.reindex(all_days, fill_value=0.0).rolling(window_days, min_periods=1).sum()
    n = rolled["n"]
    var = (rolled["ss"] - rolled["s"] ** 2 / n.where(n > 0)) / (n - 1).where(n > 1)
    std = np.sqrt(var.clip(lower=0))
    std[n < window_days * 24] = np.nan  # need at least half the periods
    share = (rolled["neg"] / n.where(n > 0)).where(n >= window_days * 24)
    return _shift_daily(std, lag_days, delivery), _shift_daily(share, lag_days, delivery)


def day_type_codes(
    delivery: pd.Series, day_of_week: pd.Series, holidays: set[dt.date]
) -> np.ndarray:
    """0 weekday, 1 weekend, 2 bank holiday, 3 Christmas period (later codes take priority)."""
    codes = np.where(day_of_week.to_numpy() >= 5, 1, 0)
    unique = delivery.unique()
    holiday = delivery.map({d: d in holidays for d in unique}).to_numpy(dtype=bool)
    christmas = delivery.map({d: is_christmas_period(d) for d in unique}).to_numpy(dtype=bool)
    return np.where(christmas, 3, np.where(holiday, 2, codes))


# --- building the exploration frame ----------------------------------------------------------


def add_exploration_features(
    data: pd.DataFrame,
    *,
    weather: pd.DataFrame | None = None,
    solar: pd.DataFrame | None = None,
    holidays: set[dt.date] | None = None,
    hour: int = 10,
) -> pd.DataFrame:
    """Add every :data:`CATALOGUE` variant column to the model dataset.

    Args:
        data: Output of :func:`~gb_price_sandbox.features.dataset.build_dataset` (complete,
            regular half-hourly UTC grid).
        weather: Half-hourly ``solar_irradiance_wm2`` and ``temperature_c`` on ``data``'s grid
            (see :func:`~gb_price_sandbox.data.open_meteo.load_weather`). Missing -> NaN columns.
        solar: Half-hourly ``solar_mw`` (PV_Live outturn). Missing -> NaN columns.
        holidays: Bank holiday dates. Missing -> no holidays flagged.
        hour: Forecast issue hour (London time, D-1).

    Returns:
        A copy of ``data`` with the extra columns.
    """
    out = data.copy()
    delivery = pd.Series(out["delivery_day"].to_numpy(), index=out.index)
    price = out["price"]
    median, mad = robust_centre(price)

    def asinh(col: pd.Series | np.ndarray) -> np.ndarray:
        return asinh_scale(col, median, mad)

    # Target
    out["price_asinh"] = asinh(price)

    # Demand and wind (NESO forecasts)
    out["demand_gw"] = out["demand_mw"] / 1000
    out["demand_z"] = (out["demand_mw"] - out["demand_mw"].mean()) / out["demand_mw"].std()
    out["wind_gw"] = out["wind_mw"] / 1000
    out["wind_share"] = out["wind_mw"] / out["demand_mw"]
    out["residual_gw"] = out["residual_demand_mw"] / 1000

    # Price lags (2, 3, 7 from the dataset; the rest of 2-14 added here for the lag profile)
    for k in LAG_PROFILE_DAYS:
        if f"price_lag_{k}d" not in out:
            out[f"price_lag_{k}d"] = price.shift(48 * k)
    for k in (2, 3, 7, 14):
        out[f"price_lag_{k}d_asinh"] = asinh(out[f"price_lag_{k}d"])
    out["price_mean_d2_asinh"] = asinh(out["price_mean_d2"])

    # Calendar
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(float)
    local = out.index.tz_convert(LONDON)
    doy = local.day_of_year.to_numpy()
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["month"] = local.month.to_numpy() - 1
    holidays = holidays or set()
    out["day_type"] = day_type_codes(delivery, out["day_of_week"], holidays)
    out["is_holiday"] = (out["day_type"] >= 2).astype(float)

    # D-1 morning prices
    am_mean, am_last = d1_morning_prices(price, delivery, hour=hour)
    out["price_d1_am_mean"] = am_mean
    out["price_d1_am_mean_asinh"] = asinh(am_mean)
    out["price_d1_last"] = am_last

    # D-2 price shape
    g = price.groupby(delivery.to_numpy())
    out["price_d2_max"] = _shift_daily(g.max(), 2, delivery)
    out["price_d2_min"] = _shift_daily(g.min(), 2, delivery)
    out["price_d2_spread"] = out["price_d2_max"] - out["price_d2_min"]

    # Recent volatility, window ending D-2
    std, neg = trailing_daily_stats(price, delivery, VOLATILITY_DAYS, 2)
    out["price_vol_7d"] = std
    out["price_neg_share_7d"] = neg
    std_z, _ = trailing_daily_stats(out["price_asinh"], delivery, VOLATILITY_DAYS, 2)
    out["price_vol_7d_asinh"] = std_z

    # Residual-demand shape (all from D's own forecasts, known at issue time)
    resid = out["residual_gw"]
    out["residual_ramp_gw"] = resid.diff()
    rg = resid.groupby(delivery.to_numpy())
    out["residual_day_max_gw"] = rg.transform("max").to_numpy()
    out["residual_day_min_gw"] = rg.transform("min").to_numpy()

    # Weather (Open-Meteo short-lead forecasts)
    w = weather.reindex(out.index) if weather is not None else pd.DataFrame(index=out.index)
    out["solar_irradiance_wm2"] = w.get("solar_irradiance_wm2", np.nan)
    out["temperature_c"] = w.get("temperature_c", np.nan)
    out["heating_degrees"] = (HEATING_BASE_C - out["temperature_c"]).clip(lower=0)

    # Solar outturn (PV_Live)
    s = solar.reindex(out.index) if solar is not None else pd.DataFrame(index=out.index)
    solar_gw = s.get("solar_mw", pd.Series(np.nan, index=out.index)) / 1000
    out["solar_outturn_gw"] = solar_gw
    out["solar_lag_2d_gw"] = solar_gw.shift(48 * 2)

    return out


# --- the catalogue ---------------------------------------------------------------------------

_P = "£/MWh"
_Z = "asinh scale"


def _lag_variants() -> tuple[Variant, ...]:
    """Every lag in the profile: the used ones shown, the rest kept for the matrix."""
    out = []
    for k in LAG_PROFILE_DAYS:
        used = k in USED_LAG_DAYS
        shown = used or k == 14
        out.append(
            Variant(
                f"price_lag_{k}d",
                f"Price {k} days earlier",
                f"Lag {k}d",
                _P,
                primary=k in (2, 7, 14),
                status="used" if used else "candidate",
                show=shown,
                in_matrix=shown,
            )
        )
        if f"price_lag_{k}d_asinh" in _ASINH_LAGS:
            out.append(
                Variant(
                    f"price_lag_{k}d_asinh",
                    f"asinh price {k} days earlier",
                    f"Lag {k}d (asinh)",
                    _Z,
                    target="price_asinh",
                    status="used" if used else "candidate",
                    show=False,
                )
            )
    return tuple(out)


_ASINH_LAGS = {f"price_lag_{k}d_asinh" for k in (2, 3, 7, 14)}

CATALOGUE: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        key="price",
        links=(MID,),
        group="Target",
        title="Price (the target)",
        status="target",
        source="Elexon Market Index Price (MID)",
        variants=(
            Variant("price", "Price", "Price", _P, primary=True),
            Variant(
                "price_asinh", "asinh-transformed price", "Price (asinh)", _Z, target="price_asinh"
            ),
        ),
        summary="What every model is trying to predict: the half-hourly Market Index Price.",
        explanation="""
GB half-hourly prices are **heavy-tailed**: most periods sit within a few tens of pounds of the
typical price, but a few spike to many times that, and some go below zero. A model fitted by
least squares on raw prices spends most of its effort on the spikes.

LEAR therefore models a transformed price: `asinh((price - median) / MAD)`, where MAD is the
median absolute deviation. `asinh` behaves like a straight line near zero and like a logarithm
further out, on both sides. So it squashes spikes without breaking on negative prices (a log
transform would). Compare the two histograms: the transformed one is far closer to symmetric.
LEAR applies the same transform to every past-price input.

The median and MAD here are taken over the whole period, for display. Inside the backtest, LEAR
takes them from each training window only.
""",
        timing="The target itself. Same-day prices are never a feature.",
    ),
    FeatureSpec(
        key="demand_wind",
        links=(NDF, WINDFOR),
        group="Used by a model today",
        title="Demand and wind",
        status="used",
        source="NESO demand (NDF) and wind (WINDFOR) forecasts, via Elexon",
        models=("LEAR",),
        variants=(
            Variant("demand_gw", "Demand forecast", "Demand", "GW", primary=True),
            Variant("wind_gw", "Wind forecast", "Wind", "GW", primary=True),
            Variant(
                "residual_gw",
                "Residual demand (demand minus wind)",
                "Residual",
                "GW",
                primary=True,
                status="built",
            ),
            Variant(
                "wind_share", "Wind as a share of demand", "Wind share", "share", status="candidate"
            ),
            Variant("demand_z", "Standardised demand", "Demand (z)", "std devs", show=False),
        ),
        summary="NESO's day-ahead forecasts of national demand and wind output, and what's left "
        "for the rest of the fleet.",
        explanation="""
**Demand** sets how far up the merit order the market has to go: higher demand needs a more
expensive plant at the margin. NDF is *transmission* demand, so rooftop solar and small wind
farms on local networks show up as lower demand. That means the demand forecast already
carries some solar information.

**Wind** has almost no running cost, so more wind pushes expensive gas plant out of the merit
order. GB has one of the largest offshore wind fleets in the world, and wind is the main source
of day-to-day price swings. Capacity has grown through the period (Dogger Bank, Moray West), so
the same GW means a smaller share of the fleet over time.

**Residual demand** (demand minus wind) is what the dispatchable fleet, mostly gas, has to
cover. It is the closest single number to "how far up the merit order are we", so it
correlates with price more strongly than either input alone. It is in the dataset but **LEAR
doesn't use it**: in a linear model, demand minus wind is already expressed by the separate
coefficients. Tree models such as LightGBM can't add features together, so for them it's a
genuinely new input.

**Wind share of demand** is often more informative than the wind level. 10 GW of wind on a 40 GW
winter evening displaces some gas; the same 10 GW on a 20 GW summer night can push prices
negative.

LEAR also **standardises** demand and wind (subtract the mean, divide by the standard
deviation) so the LASSO penalty treats every input on the same scale. That doesn't change the
shape, so it is only in the correlation matrix (switch on transformed versions).
""",
        timing="The latest version of each forecast published by 10:00 on D-1. NESO revises both "
        "many times; later revisions are never used. Wind is interpolated from hourly to "
        "half-hourly.",
    ),
    FeatureSpec(
        key="past_prices",
        links=(MID,),
        group="Used by a model today",
        title="Past prices",
        status="used",
        source="Elexon MID",
        models=("LEAR", "Seasonal naive"),
        lag_profile=True,
        timeseries=False,
        variants=(
            *_lag_variants(),
            Variant("price_mean_d2", "Average price on D-2", "Mean D-2", _P, primary=True),
            Variant(
                "price_mean_d2_asinh",
                "asinh of the D-2 average",
                "Mean D-2 (asinh)",
                _Z,
                target="price_asinh",
                show=False,
            ),
            Variant(
                "price_d1_am_mean",
                "Average price, D-1 00:00 to 09:00",
                "D-1 am mean",
                _P,
                primary=True,
                status="candidate",
            ),
            Variant(
                "price_d1_am_mean_asinh",
                "asinh of the D-1 morning average",
                "D-1 am (asinh)",
                _Z,
                target="price_asinh",
                status="candidate",
                show=False,
            ),
            Variant(
                "price_d1_last",
                "Last price known at issue (D-1 08:30)",
                "D-1 last",
                _P,
                status="candidate",
            ),
        ),
        summary="The price itself, shifted back in time: the same half-hour some days earlier, "
        "and recent averages.",
        explanation="""
Prices are **persistent**: gas prices, plant availability and the weather pattern change slowly,
so the recent past is a good guide. Every past-price feature is the same price series, *lagged*:
the price in the same half-hour exactly *k* × 24 hours earlier.

**Which lags are allowed?** The forecast is made at 10:00 on D-1, when most of D-1 hasn't
happened. So the shortest full lag is **2 days**. The chart above shows how the correlation
with price fades with the lag and **ticks back up at 7 and 14 days**: a Monday is most like
previous Mondays. That is why the models use 2, 3 and 7 days. The seasonal-naive baseline is
just the 7-day lag on its own.

The lags are highly correlated with one another (see the matrix below). LASSO handles this by
shrinking the ones that add nothing, so LEAR doesn't need to be told which to drop.

**Averages.** The **D-2 average** strips out the within-day shape and tracks the overall
**price level**, which mostly follows the gas price. It stands in for a gas price feature, which
this project has no data source for. The **D-1 morning average** (a candidate) is the part of
yesterday that *had* happened by 10:00: a full day more recent than anything else the models
see, though it is overnight and early-morning prices, with their own level.

LEAR uses all of these on the **asinh scale** (see the Price page). Switch on transformed
versions in the matrix to see them.
""",
        timing="Lags of 2 days or more and the D-2 average use fully settled days. The D-1 "
        f"morning features use periods ending by 09:00, allowing {PUBLICATION_MARGIN_H} hour "
        "for Elexon to publish them before the 10:00 issue time. Tested for leakage.",
    ),
    FeatureSpec(
        key="calendar",
        derived_note="Time of day, day of week and season are worked out from the timestamps "
        "themselves.",
        links=(BANK_HOLIDAYS,),
        group="Used by a model today",
        title="Calendar",
        status="used",
        source="Calendar; GOV.UK bank holidays (England and Wales)",
        models=("LEAR",),
        variants=(
            Variant(
                "period_of_day",
                "Half-hour of the day (local time)",
                "Half-hour",
                kind="categorical",
                in_matrix=False,
                categories=PERIOD_LABELS,
            ),
            Variant(
                "day_of_week",
                "Day of the week",
                "Day of week",
                kind="categorical",
                in_matrix=False,
                categories=DAY_NAMES,
            ),
            Variant(
                "day_type",
                "Type of day",
                "Day type",
                kind="categorical",
                in_matrix=False,
                categories=DAY_TYPES,
                status="candidate",
            ),
            Variant(
                "month",
                "Month",
                "Month",
                kind="categorical",
                in_matrix=False,
                categories=MONTH_NAMES,
                status="candidate",
            ),
            Variant(
                "is_weekend",
                "Weekend",
                "Weekend",
                kind="categorical",
                primary=True,
                categories=("Weekday", "Weekend"),
                show=False,
            ),
            Variant(
                "is_holiday",
                "Bank holiday or Christmas period",
                "Holiday",
                kind="categorical",
                primary=True,
                categories=("No", "Yes"),
                status="candidate",
                show=False,
            ),
            Variant(
                "doy_cos",
                "cos(day of year): +1 in winter, -1 in summer",
                "Season (cos)",
                primary=True,
                status="candidate",
                show=False,
            ),
            Variant(
                "doy_sin",
                "sin(day of year): +1 in spring, -1 in autumn",
                "Season (sin)",
                status="candidate",
                show=False,
            ),
        ),
        summary="Time of day, day of week, holidays and season: patterns known years in advance.",
        explanation="""
**Half-hour of the day.** The most regular pattern in the data: low overnight, a morning ramp,
a midday dip as solar peaks, and the **evening peak** around 17:00 to 19:00 when demand is high
and solar has gone. LEAR doesn't use the half-hour as an input. Instead it fits **48 separate
models**, one per half-hour, so each gets its own coefficients: wind can matter more overnight
than at the evening peak, for example.

**Day of the week.** Demand is lower at weekends, so prices fall, and most negative prices
happen then. A linear model can't use "day 0 to 6" as a number, since Sunday isn't six times
Monday, so LEAR uses **one-hot dummies**: six 0/1 columns with Monday as the baseline.

**Holidays** (candidate). On a bank holiday demand looks like a Sunday's, but a model using only
the day of the week forecasts it as a weekday, too high. The Christmas period (24 December to
1 January) is lower still. There are only about 8 bank holidays a year, so one "holiday" flag
works better than one per holiday. England and Wales dates are used; Scotland's differ slightly.

**Season** (candidate). Prices are higher in winter (more demand, dearer gas) and lower in late
spring and summer. Day-of-year as a plain number would put 31 December and 1 January at opposite
ends, so models use a **sine and cosine** pair that wraps the year round a circle. Over 2022 to
2023 the seasonal pattern is swamped by the gas crisis, so treat month averages with care.

In the correlation matrix these appear as 0/1 flags (weekend, holiday) and the season cosine,
since a correlation with "day 0 to 6" means nothing.
""",
        timing="The calendar is known in advance; GOV.UK publishes bank holidays years ahead.",
    ),
    FeatureSpec(
        key="solar",
        links=(OPEN_METEO, PV_LIVE),
        group="Candidates",
        title="Solar",
        status="candidate",
        source="Open-Meteo historical forecasts (irradiance); Sheffield Solar PV_Live (outturn)",
        variants=(
            Variant(
                "solar_irradiance_wm2",
                "Irradiance forecast (GB solar-weighted)",
                "Irradiance fc",
                "W/m²",
                primary=True,
            ),
            Variant(
                "solar_outturn_gw",
                "Solar generation (PV_Live)",
                "Solar outturn",
                "GW",
                known_at_issue=False,
            ),
            Variant("solar_lag_2d_gw", "Solar generation 2 days earlier", "Solar lag 2d", "GW"),
        ),
        summary="Sunshine forecast, and how much solar the GB fleet actually produced.",
        explanation="""
GB has around 20 GW of solar, most of it on rooftops and small farms connected to local
networks. It doesn't bid into the market; it shows up as **lower demand** at midday. On sunny
days, especially at weekends, it pushes midday prices down and sometimes below zero. The
Forecast sandbox shows midday is where the current models' errors are largest, which is why
this is the first candidate to add.

- The **irradiance forecast** is the sunlight forecast, averaged over sites weighted roughly by
  where solar is installed (mostly the south of England).
- **PV_Live outturn** is the estimated actual solar output. A forecaster doesn't have this for
  day D, so it is shown only as a reference: compare it with the forecast to see how good the
  forecast is.
- **Solar 2 days earlier** is known at issue time. It tracks the season and recent cloudiness,
  but not tomorrow's weather.

Over the whole day the correlation with price is close to zero, which badly understates solar's
importance: at night it is zero whatever the price. **Use the time-of-day filter at the top of
the page** to look at 10:00 to 15:00 only, and the negative relationship appears. Restrict the
years to 2024 onwards and it is clearer still, as capacity has grown and the gas-crisis years
drop out. A model per half-hour (LEAR) or a tree model picks up this interaction automatically.
""",
        timing="**Irradiance:** Open-Meteo's archive keeps the first hours of each forecast run, "
        "so this is a same-day forecast, better than one made on D-1. A backtest would need "
        "true D-1 vintages (Open-Meteo's previous-runs archive, from 2024 only). "
        "**Outturn:** not known at issue time. **2-day lag:** known.",
    ),
    FeatureSpec(
        key="price_shape",
        links=(MID,),
        group="Candidates",
        title="Recent price shape and volatility",
        status="candidate",
        source="Elexon MID",
        variants=(
            Variant("price_d2_spread", "D-2 spread (max - min)", "D-2 spread", _P, primary=True),
            Variant("price_d2_max", "D-2 highest half-hour", "D-2 max", _P),
            Variant("price_d2_min", "D-2 lowest half-hour", "D-2 min", _P),
            Variant(
                "price_vol_7d",
                "Price standard deviation, 7 days to D-2",
                "Volatility 7d",
                _P,
                primary=True,
            ),
            Variant(
                "price_vol_7d_asinh",
                "asinh-price standard deviation, 7 days to D-2",
                "Volatility 7d (asinh)",
                _Z,
            ),
            Variant(
                "price_neg_share_7d",
                "Share of negative half-hours, 7 days to D-2",
                "Neg. share 7d",
                "share",
            ),
        ),
        summary="How peaky D-2 was, and how jumpy prices have been over the past week.",
        explanation="""
**D-2 shape.** The published LEAR model uses the full 24-hour price profile of each lagged day;
this project uses only the same half-hour and the daily mean. The D-2 peak, trough and spread
are a cheap way to add the missing shape. The spread matters most for a battery, which earns
roughly the gap between the cheapest and dearest hours.

**Volatility.** Volatility tends to cluster: a stormy week is usually followed by more of the
same. For a point forecast this barely helps. For the **quantile models** it should help a lot,
because it tells them how wide the 10% to 90% band ought to be. The asinh version isn't
dominated by one or two spikes, so it measures how jumpy prices typically were. The **share of
negative half-hours** captures how often the system has recently been oversupplied, which is
when a battery charges for free.
""",
        timing="Uses delivery days D-8 to D-2, all settled by 10:00 on D-1.",
    ),
    FeatureSpec(
        key="residual_shape",
        links=(NDF, WINDFOR),
        group="Candidates",
        title="Residual-demand shape",
        status="candidate",
        source="NESO demand and wind forecasts",
        variants=(
            Variant(
                "residual_ramp_gw",
                "Change in residual demand from the previous half-hour",
                "Residual ramp",
                "GW",
                primary=True,
            ),
            Variant("residual_day_max_gw", "Daily peak residual demand", "Residual peak", "GW"),
            Variant(
                "residual_day_min_gw", "Daily minimum residual demand", "Residual trough", "GW"
            ),
        ),
        summary="How fast the residual load ramps, and the day's peak and trough.",
        explanation="""
Prices react to **ramps** as well as levels. When residual demand rises quickly, for example as
solar fades into the evening peak, flexible plant has to start up and prices jump. When it falls
quickly, plant that can't turn down fast enough has to be paid to stay on.

The daily peak and trough describe the whole of D in two numbers. A big gap between them means
a peaky day, and probably a good one for a battery.

**Interconnector flows** would belong here too. GB imports and exports up to about 10 GW, but
Elexon only publishes actual flows, not forecasts, so they are left out.
""",
        timing="Built from D's own demand and wind forecasts (D-1 10:00 vintages).",
    ),
    FeatureSpec(
        key="temperature",
        links=(OPEN_METEO,),
        group="Candidates",
        title="Temperature",
        status="candidate",
        source="Open-Meteo historical forecasts",
        variants=(
            Variant(
                "temperature_c",
                "Temperature forecast (GB population-weighted)",
                "Temperature",
                "°C",
                primary=True,
            ),
            Variant(
                "heating_degrees",
                "Heating degrees (15.5 °C minus temperature, floored at 0)",
                "Heating deg.",
                "°C",
            ),
        ),
        summary="Forecast air temperature, averaged over the big population centres.",
        explanation="""
Cold weather raises electricity demand (electric heating, longer evenings indoors) and, more
importantly, **gas demand for heating**. That pushes up gas prices and so power prices. GB
has little air conditioning, so hot weather matters much less, although heat waves reduce
thermal plant efficiency and interconnector imports.

**Heating degrees** encode this one-sided effect: zero above 15.5 °C, then rising as it gets
colder. Most of the effect on electricity demand is already in NESO's demand forecast. The
remaining value is mainly through gas.
""",
        timing="Same caveat as irradiance: Open-Meteo's archive is a short-lead forecast, a little "
        "better than a D-1 forecast would be.",
    ),
)

CATALOGUE_BY_KEY: dict[str, FeatureSpec] = {f.key: f for f in CATALOGUE}


def ordered_catalogue() -> list[FeatureSpec]:
    """Catalogue in selector order: the target, features used today, then candidates."""
    return list(CATALOGUE)


def all_variants() -> list[Variant]:
    """Every variant in the catalogue."""
    return [v for f in CATALOGUE for v in f.variants]


def lag_profile(frame: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Correlation of price with its own value ``k`` days earlier, for each profile lag.

    Uses the precomputed ``price_lag_{k}d`` columns, so it works on filtered frames.

    Returns:
        One row per lag with ``lag_days``, ``correlation`` and ``used`` (a model uses it).
    """
    rows = [
        {
            "lag_days": k,
            "correlation": float(frame[f"price_lag_{k}d"].corr(frame["price"], method=method)),
            "used": k in USED_LAG_DAYS,
        }
        for k in LAG_PROFILE_DAYS
    ]
    return pd.DataFrame(rows)


# --- summaries for the charts (pure data, unit-testable) -------------------------------------


def daily_means(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-delivery-day mean of ``columns`` (index: datetime64 days)."""
    out = frame.groupby("delivery_day")[columns].mean()
    out.index = pd.to_datetime(out.index)
    return out


def binned_relationship(x: pd.Series, y: pd.Series, bins: int = 20) -> pd.DataFrame:
    """Median and 10th/90th percentiles of ``y`` within quantile bins of ``x``.

    Robust to the price spikes that make a raw scatter of 70,000 points unreadable.

    Returns:
        One row per bin with ``x`` (median of x in the bin), ``p10``, ``p50``, ``p90``, ``n``.
    """
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if x.nunique() < 2:
        return pd.DataFrame(columns=["x", "p10", "p50", "p90", "n"])
    # Bin on the values, merging bins whose edges coincide: a feature that is often exactly
    # zero (solar at night) gets one bin at zero rather than several stacked on top of it.
    codes = pd.qcut(x, q=min(bins, x.nunique()), labels=False, duplicates="drop")
    g = pd.DataFrame({"x": x.to_numpy(), "y": y.to_numpy(), "bin": codes.to_numpy()}).groupby("bin")
    return pd.DataFrame(
        {
            "x": g["x"].median(),
            "p10": g["y"].quantile(0.1),
            "p50": g["y"].median(),
            "p90": g["y"].quantile(0.9),
            "n": g["y"].size(),
        }
    ).reset_index(drop=True)


def by_category(codes: pd.Series, y: pd.Series, labels: tuple[str, ...]) -> pd.DataFrame:
    """Mean, median and 10th/90th percentiles of ``y`` for each category code.

    Returns:
        One row per category present, in code order, indexed by label.
    """
    ok = codes.notna() & y.notna()
    g = y[ok].groupby(codes[ok].astype(int))
    out = pd.DataFrame(
        {
            "mean": g.mean(),
            "p10": g.quantile(0.1),
            "p50": g.median(),
            "p90": g.quantile(0.9),
            "n": g.size(),
        }
    ).sort_index()
    out.index = [labels[i] for i in out.index]
    return out


def correlation_with(
    frame: pd.DataFrame, columns: list[str], target: str, method: str
) -> pd.Series:
    """Correlation of each column with ``target`` (pairwise complete)."""
    return frame[columns].corrwith(frame[target], method=method)


def correlation_matrix(
    frame: pd.DataFrame, columns: list[str], method: str = "spearman"
) -> pd.DataFrame:
    """Pairwise-complete correlation matrix of ``columns``.

    Args:
        frame: Exploration frame.
        columns: Columns to include (columns with no data are dropped).
        method: ``pearson`` (straight-line) or ``spearman`` (rank, robust to spikes and
            unchanged by monotone transforms such as asinh).
    """
    present = [c for c in columns if frame[c].notna().any()]
    sub = frame[present]
    if method == "spearman":
        # Rank once, then Pearson on ranks: much faster than pandas' per-pair Spearman. Ranks
        # are over each column's own non-missing rows, a negligible difference here.
        sub = sub.rank()
        method = "pearson"
    return sub.corr(method=method)


def matrix_columns(include_transforms: bool) -> list[Variant]:
    """Variants shown in the correlation matrix: the primary ones, or all matrix variants."""
    return [v for v in all_variants() if v.in_matrix and (include_transforms or v.primary)]


#: If at least this share of values is exactly zero, the zeros are reported separately and
#: the histogram shows the rest (otherwise night-time solar would be one giant bar).
ZERO_SPIKE_SHARE = 0.2


def histogram(
    values: pd.Series, bins: int = 60, clip: float = 0.005
) -> tuple[pd.DataFrame, float, float]:
    """Histogram of ``values`` over its central range, so a few spikes don't set the scale.

    Args:
        values: Values to bin (NaN ignored).
        bins: Number of equal-width bins.
        clip: Share cut from each tail before binning.

    Returns:
        ``(table, outside, zeros)``: one row per bin (``left``, ``right``, ``mid``, ``share``),
        the share of values outside the plotted range, and the share of exact zeros left out
        of the bins (0 unless they make up at least :data:`ZERO_SPIKE_SHARE`). Shares are of
        all non-missing values.
    """
    v = values.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return pd.DataFrame(columns=["left", "right", "mid", "share"]), 0.0, 0.0
    total = len(v)
    zeros = float(np.mean(v == 0))
    if zeros >= ZERO_SPIKE_SHARE and zeros < 1:
        v = v[v != 0]
    else:
        zeros = 0.0
    lo, hi = np.quantile(v, [clip, 1 - clip])
    if hi <= lo:
        lo, hi = float(v.min()) - 0.5, float(v.max()) + 0.5
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    inside = counts.sum()
    table = pd.DataFrame(
        {
            "left": edges[:-1],
            "right": edges[1:],
            "mid": (edges[:-1] + edges[1:]) / 2,
            "share": counts / total,
        }
    )
    return table, float((len(v) - inside) / total), zeros
