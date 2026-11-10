"""Figures for the Battery optimiser tab."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gb_price_sandbox.timing import LONDON
from gb_price_sandbox.viz.theme import palette, style


def fig_capture_rates(table: pd.DataFrame, colors: dict[str, str], dark: bool) -> go.Figure:
    """Capture rate per model as horizontal bars, labelled with the rate and the revenue."""
    pal = palette(dark)
    t = table.sort_values("Capture rate")
    fig = go.Figure(
        go.Bar(
            x=t["Capture rate"] * 100,
            y=t.index,
            orientation="h",
            marker={"color": [colors[n] for n in t.index], "line": {"width": 0}},
            text=[
                f"{r:.0%} · £{v:,.0f}"
                for r, v in zip(t["Capture rate"], t["Revenue (GBP)"], strict=True)
            ],
            textposition="outside",
            textfont={"color": pal.ink_secondary},
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}% of perfect-foresight revenue<extra></extra>",
        )
    )
    fig.update_layout(bargap=0.5)
    fig.update_xaxes(
        title="Capture rate (% of perfect-foresight revenue)",
        range=[min(0, t["Capture rate"].min() * 100), 125],
        ticksuffix="%",
        tickvals=[0, 20, 40, 60, 80, 100],
    )
    return style(fig, dark, height=80 + 44 * len(t), legend=False, grid="x")


def fig_dispatch_day(
    schedules: dict[str, pd.DataFrame],
    forecasts: dict[str, pd.Series],
    names: list[str],
    colors: dict[str, str],
    day: dt.date,
    dark: bool,
) -> go.Figure:
    """One day: prices (top) and state of charge (bottom), stacked panels sharing the time axis.

    Args:
        schedules: Per-period dispatch by name (must include the perfect-foresight schedule).
        forecasts: Median forecast prices by model name (drawn in the price panel).
        names: Schedules to draw, in order.
        colors: Fixed colour per name (including ``"Actual"``).
        day: Delivery day to show.
        dark: Dark mode.
    """
    first = next(iter(schedules.values()))
    local = first.index.tz_convert(LONDON)
    mask = local.date == day
    x = local[mask]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("Price (£/MWh)", "State of charge (MWh)"),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=first.loc[mask, "settlement_price"],
            name="Actual price",
            mode="lines",
            line={"width": 2, "color": colors["Actual"], "shape": "hv"},
            legendgroup="Actual",
            hovertemplate="Actual £%{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for name in names:
        if name in forecasts:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=forecasts[name][mask],
                    name=name,
                    mode="lines",
                    line={"width": 2, "color": colors[name], "shape": "hv"},
                    legendgroup=name,
                    opacity=0.7,
                    hovertemplate=f"{name} forecast £%{{y:.0f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        sched = schedules[name]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sched.loc[mask, "soc_mwh"],
                name=name,
                mode="lines",
                line={"width": 2, "color": colors[name]},
                legendgroup=name,
                showlegend=name not in forecasts,
                hovertemplate=f"{name} SoC %{{y:.2f}} MWh<extra></extra>",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(hovermode="x unified")
    fig = style(fig, dark, height=540)
    fig.update_layout(margin_t=72, legend_y=1.1)  # clear of the top panel's title
    pal = palette(dark)
    fig.update_annotations(font={"color": pal.ink_secondary, "size": 13}, x=0, xanchor="left")
    fig.update_yaxes(rangemode="tozero", row=2, col=1)
    return fig
