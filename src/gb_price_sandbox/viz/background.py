"""Data summaries and figures for the Background tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gb_price_sandbox.merit_order import Block as StackBlock
from gb_price_sandbox.merit_order import Clearing, stack_frame
from gb_price_sandbox.timing import LONDON
from gb_price_sandbox.viz.theme import palette, style

# --- summaries (pure data, unit-testable) ----------------------------------------------------


def daily_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Per delivery day: mean/min/max price, mean demand and wind forecasts (GW)."""
    g = data.groupby("delivery_day")
    out = pd.DataFrame(
        {
            "price_mean": g["price"].mean(),
            "price_min": g["price"].min(),
            "price_max": g["price"].max(),
            "demand_gw": g["demand_mw"].mean() / 1000,
            "wind_gw": g["wind_mw"].mean() / 1000,
        }
    )
    out.index = pd.to_datetime(out.index)
    return out


def headline_stats(data: pd.DataFrame) -> dict[str, object]:
    """Numbers quoted in the page text, computed rather than hard-coded."""
    daily = daily_summary(data)
    yearly = data.groupby(data.index.tz_convert(LONDON).year)["price"].mean()
    peak_day = daily["price_mean"].idxmax()
    return {
        "first_day": daily.index.min().date(),
        "last_day": daily.index.max().date(),
        "yearly_mean": yearly.round(0).to_dict(),
        "peak_day": peak_day.date(),
        "peak_day_mean": float(daily.loc[peak_day, "price_mean"]),
        "max_period_price": float(data["price"].max()),
        "min_period_price": float(data["price"].min()),
        "negative_hours": float((data["price"] < 0).sum() / 2),
    }


def negative_hours_by_month(data: pd.DataFrame) -> pd.Series:
    """Hours with a negative MID price, per calendar month (local time)."""
    local = data.index.tz_convert(LONDON)
    month = pd.PeriodIndex(local.tz_localize(None), freq="M")
    neg = (data["price"] < 0).astype(float) / 2
    out = neg.groupby(month).sum()
    out.index = out.index.to_timestamp()
    return out


def daily_shape_by_year(data: pd.DataFrame) -> pd.DataFrame:
    """Mean price by local half-hour of day (rows, 0-47) for each year (columns)."""
    year = data.index.tz_convert(LONDON).year
    frame = pd.DataFrame(
        {"period_of_day": data["period_of_day"].to_numpy(), "year": year, "price": data["price"]}
    )
    return frame.pivot_table(index="period_of_day", columns="year", values="price", aggfunc="mean")


# --- figures ---------------------------------------------------------------------------------


def fig_price_history(daily: pd.DataFrame, dark: bool) -> go.Figure:
    """Daily mean price (faint) under its 30-day rolling mean (bold).

    Daily min/max go in the tooltip rather than a band: a few extreme half-hours would
    otherwise set the y-scale and flatten everything else.
    """
    pal = palette(dark)
    color = pal.series[0]
    rolling = daily["price_mean"].rolling(30, min_periods=15, center=True).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["price_mean"],
            mode="lines",
            line={"width": 1, "color": _alpha(color, 0.35)},
            name="Daily mean",
            customdata=np.stack([daily["price_min"], daily["price_max"]], axis=1),
            hovertemplate="%{x|%a %d %b %Y}<br>Daily mean £%{y:.0f}/MWh"
            "<br>Half-hours from £%{customdata[0]:.0f} to £%{customdata[1]:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=rolling,
            mode="lines",
            line={"width": 2.5, "color": color},
            name="30-day average",
            hovertemplate="30-day average £%{y:.0f}/MWh<extra></extra>",
        )
    )
    fig.update_yaxes(title="£/MWh", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return style(fig, dark, height=400)


def fig_negative_hours(monthly: pd.Series, dark: bool) -> go.Figure:
    """Bars of negative-price hours per month."""
    pal = palette(dark)
    fig = go.Figure(
        go.Bar(
            x=monthly.index,
            y=monthly.to_numpy(),
            marker={"color": pal.series[0]},
            marker_line_width=0,
            hovertemplate="%{x|%b %Y}: %{y:.0f} hours<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.2)
    fig.update_yaxes(title="Hours below £0/MWh")
    return style(fig, dark, height=300, legend=False)


def fig_daily_shape(shape: pd.DataFrame, dark: bool) -> go.Figure:
    """Average price through the day, one line per year, direct-labelled at the evening peak."""
    pal = palette(dark)
    labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    fig = go.Figure()
    for i, year in enumerate(shape.columns):
        color = pal.series[i % len(pal.series)]
        y = shape[year]
        fig.add_trace(
            go.Scatter(
                x=[labels[p] for p in shape.index],
                y=y,
                mode="lines",
                name=str(year),
                line={"width": 2, "color": color},
                hovertemplate=f"{year} %{{x}}: £%{{y:.0f}}/MWh<extra></extra>",
            )
        )
        peak = y.idxmax()
        fig.add_annotation(
            x=labels[peak],
            y=y[peak],
            text=str(year),
            showarrow=False,
            yshift=10,
            font={"color": pal.ink_secondary, "size": 11},
        )
    fig.update_xaxes(title="Time of day (UK)", tickvals=labels[::8])
    fig.update_yaxes(title="Average £/MWh")
    fig.update_layout(hovermode="x unified")
    return style(fig, dark, height=380)


def fig_demand_wind(daily: pd.DataFrame, dark: bool) -> go.Figure:
    """Monthly means of the day-ahead demand and wind forecasts (same unit, one axis)."""
    pal = palette(dark)
    monthly = daily[["demand_gw", "wind_gw"]].resample("MS").mean()
    fig = go.Figure()
    for i, (col, name) in enumerate([("demand_gw", "Demand"), ("wind_gw", "Wind")]):
        fig.add_trace(
            go.Scatter(
                x=monthly.index,
                y=monthly[col],
                mode="lines",
                name=name,
                line={"width": 2, "color": pal.series[i]},
                hovertemplate=f"{name} %{{x|%b %Y}}: %{{y:.1f}} GW<extra></extra>",
            )
        )
    fig.update_yaxes(title="GW (monthly mean of day-ahead forecasts)", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return style(fig, dark, height=340)


def fig_price_vs_residual(daily: pd.DataFrame, dark: bool) -> go.Figure:
    """Daily price against residual demand (demand - wind): the merit order, seen in the data.

    Two groups only (the 2022 gas crisis vs later), so colour stays readable.
    """
    pal = palette(dark)
    d = daily.dropna(subset=["price_mean", "demand_gw", "wind_gw"]).copy()
    d["residual_gw"] = d["demand_gw"] - d["wind_gw"]
    fig = go.Figure()
    groups = [("2022 (gas crisis)", d.index.year == 2022), ("2023 onwards", d.index.year >= 2023)]
    for i, (name, mask) in enumerate(groups):
        sub = d[mask]
        fig.add_trace(
            go.Scatter(
                x=sub["residual_gw"],
                y=sub["price_mean"],
                mode="markers",
                name=name,
                marker={
                    "size": 7,
                    "color": pal.series[i],
                    "opacity": 0.6,
                    "line": {"width": 1, "color": pal.surface},
                },
                customdata=sub.index.strftime("%d %b %Y"),
                hovertemplate="%{customdata}<br>Residual demand %{x:.1f} GW"
                "<br>£%{y:.0f}/MWh<extra>" + name + "</extra>",
            )
        )
    fig.update_xaxes(title="Residual demand: demand minus wind (GW, daily mean)")
    fig.update_yaxes(title="Daily mean price (£/MWh)")
    return style(fig, dark, height=420, grid="both")


def fig_merit_order(
    blocks: list[StackBlock], demand_gw: float, result: Clearing, dark: bool
) -> go.Figure:
    """Supply stack as bars (width = capacity, height = cost) with demand and clearing price."""
    pal = palette(dark)
    df = stack_frame(blocks)
    # A £0 block would be a zero-height (invisible) bar: give it a sliver so it shows.
    sliver = 0.015 * (df["cost_gbp_mwh"].max() - min(df["cost_gbp_mwh"].min(), 0))
    fuel = df["technology"].str.split(" ").str[0]
    fuel_order = list(dict.fromkeys(fuel))
    fuel_color = {f: pal.series[i] for i, f in enumerate(fuel_order)}
    fig = go.Figure()
    shown = set()
    for _, row in df.iterrows():
        f = row["technology"].split(" ")[0]
        fig.add_trace(
            go.Bar(
                x=[(row["start_gw"] + row["end_gw"]) / 2],
                y=[row["cost_gbp_mwh"] if abs(row["cost_gbp_mwh"]) > sliver else sliver],
                width=[max(row["capacity_gw"] - 0.15, 0.05)],
                marker={"color": fuel_color[f], "line": {"width": 0}},
                name=f,
                legendgroup=f,
                showlegend=f not in shown,
                hovertemplate=f"{row['technology']}<br>{row['capacity_gw']:.1f} GW at "
                f"£{row['cost_gbp_mwh']:.0f}/MWh<extra></extra>",
            )
        )
        shown.add(f)
    fig.add_vline(x=demand_gw, line={"color": pal.ink, "width": 2})
    fig.add_annotation(
        x=demand_gw,
        y=1,
        yref="paper",
        text=f"Demand {demand_gw:.0f} GW",
        showarrow=False,
        xanchor="left",
        xshift=6,
        font={"color": pal.ink},
    )
    if result.price_gbp_mwh is not None:
        fig.add_hline(y=result.price_gbp_mwh, line={"color": pal.ink_secondary, "width": 1})
        fig.add_annotation(
            x=0,
            xref="paper",
            y=result.price_gbp_mwh,
            text=f"Price £{result.price_gbp_mwh:.0f}",
            showarrow=False,
            xanchor="left",
            yshift=10,
            font={"color": pal.ink_secondary},
        )
    fig.update_layout(bargap=0, barmode="overlay")
    fig.update_xaxes(
        title="Cumulative capacity (GW)", range=[0, max(df["end_gw"].max(), demand_gw) + 1]
    )
    fig.update_yaxes(title="Marginal cost (£/MWh)", zeroline=True, zerolinecolor=pal.baseline)
    return style(fig, dark, height=420)


def _alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"
