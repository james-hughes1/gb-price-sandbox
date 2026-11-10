"""Figures for the Forecast sandbox tab."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go

from gb_price_sandbox.viz.theme import ACTUAL, palette, style


def fig_forecast_window(
    results: dict[str, pd.DataFrame],
    colors: dict[str, str],
    start_day: dt.date,
    days: int,
    dark: bool,
) -> go.Figure:
    """Realised price and each model's median forecast over ``days`` days from ``start_day``."""
    first = next(iter(results.values()))
    end_day = start_day + dt.timedelta(days=days)
    window = (first["delivery_day"] >= start_day) & (first["delivery_day"] < end_day)
    x = first.index[window].tz_convert("Europe/London")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=first.loc[window, "y"],
            name=ACTUAL,
            mode="lines",
            line={"width": 2, "color": colors[ACTUAL]},
            hovertemplate="Actual £%{y:.0f}<extra></extra>",
        )
    )
    for name, res in results.items():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=res.loc[window, "q50"],
                name=name,
                mode="lines",
                line={"width": 2, "color": colors[name]},
                hovertemplate=f"{name} £%{{y:.0f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title="£/MWh")
    fig.update_layout(hovermode="x unified")
    return style(fig, dark, height=400)


def fig_error_by_time_of_day(table: pd.DataFrame, colors: dict[str, str], dark: bool) -> go.Figure:
    """MAE of each model's median by local half-hour of day."""
    labels = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    fig = go.Figure()
    for name in table.columns:
        fig.add_trace(
            go.Scatter(
                x=[labels[p] for p in table.index],
                y=table[name],
                name=name,
                mode="lines",
                line={"width": 2, "color": colors[name]},
                hovertemplate=f"{name} %{{x}}: MAE £%{{y:.1f}}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Time of day (UK)", tickvals=labels[::8])
    fig.update_yaxes(title="Mean absolute error (£/MWh)", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return style(fig, dark, height=340)


def fig_metric_bars(
    board: pd.DataFrame, metric: str, colors: dict[str, str], dark: bool
) -> go.Figure:
    """Horizontal bars of one leaderboard metric, best at the top, value labels at bar ends."""
    pal = palette(dark)
    b = board.sort_values(metric, ascending=False)
    fig = go.Figure(
        go.Bar(
            x=b[metric],
            y=b.index,
            orientation="h",
            marker={"color": [colors[n] for n in b.index], "line": {"width": 0}},
            text=[f"£{v:.1f}" for v in b[metric]],
            textposition="outside",
            textfont={"color": pal.ink_secondary},
            cliponaxis=False,
            hovertemplate="%{y}: £%{x:.2f}/MWh<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.5)
    fig.update_xaxes(title=f"{metric} (£/MWh, lower is better)", rangemode="tozero")
    return style(fig, dark, height=80 + 44 * len(b), legend=False, grid="x")
