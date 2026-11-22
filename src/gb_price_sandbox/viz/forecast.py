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
    dashes: dict[str, str] | None = None,
) -> go.Figure:
    """Realised price and each model's median forecast over ``days`` days from ``start_day``.

    ``dashes`` (optional) tells apart models that share a family colour.
    """
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
                line={"width": 2, "color": colors[name], "dash": (dashes or {}).get(name, "solid")},
                hovertemplate=f"{name} £%{{y:.0f}}<extra></extra>",
            )
        )
    fig.update_yaxes(title="£/MWh")
    fig.update_layout(hovermode="x unified")
    return style(fig, dark, height=400)


def fig_metric_bars(
    board: pd.DataFrame, metric: str, colors: dict[str, str], dark: bool
) -> go.Figure:
    """Horizontal bars of one leaderboard metric, best at the top, value labels at bar ends."""
    pal = palette(dark)
    b = board.sort_values(metric, ascending=False)
    share = metric == "Coverage error"
    labels = [f"{v:.0%}" if share else f"£{v:.1f}" for v in b[metric]]
    hover = "%{y}: %{x:.1%}<extra></extra>" if share else "%{y}: £%{x:.2f}/MWh<extra></extra>"
    unit = "share of periods" if share else "£/MWh"
    fig = go.Figure(
        go.Bar(
            x=b[metric],
            y=b.index,
            orientation="h",
            marker={"color": [colors[n] for n in b.index], "line": {"width": 0}},
            text=labels,
            textposition="outside",
            textfont={"color": pal.ink_secondary},
            cliponaxis=False,
            hovertemplate=hover,
        )
    )
    fig.update_layout(bargap=0.5)
    fig.update_xaxes(title=f"{metric} ({unit}, lower is better)", rangemode="tozero")
    if share:
        fig.update_xaxes(tickformat=".0%")
    return style(fig, dark, height=80 + 44 * len(b), legend=False, grid="x")


def _alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def fig_family_window(
    results: dict[str, pd.DataFrame],
    names: list[str],
    colors: dict[str, str],
    dashes: dict[str, str],
    start_day: dt.date,
    days: int,
    dark: bool,
) -> go.Figure:
    """Actual price, plus each model's median and (where it has one) its 10-90% band.

    Bands are drawn faintly in the model's colour, under the lines, so several stay readable.
    """
    first = results[names[0]]
    end_day = start_day + dt.timedelta(days=days)
    window = (first["delivery_day"] >= start_day) & (first["delivery_day"] < end_day)
    x = first.index[window].tz_convert("Europe/London")
    fig = go.Figure()
    for name in names:
        res = results[name].loc[window]
        if (res["q90"] - res["q10"]).abs().max() > 1e-9:
            fig.add_trace(
                go.Scatter(
                    x=list(x) + list(x[::-1]),
                    y=list(res["q90"]) + list(res["q10"][::-1]),
                    fill="toself",
                    fillcolor=_alpha(colors[name], 0.15),
                    line={"width": 0},
                    name=f"{name} 10-90%",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
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
    for name in names:
        res = results[name].loc[window]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=res["q50"],
                name=name,
                mode="lines",
                line={"width": 2, "color": colors[name], "dash": dashes.get(name, "solid")},
                customdata=res[["q10", "q90"]].to_numpy(),
                hovertemplate=f"{name} £%{{y:.0f}} (10-90%: £%{{customdata[0]:.0f}} to "
                "£%{customdata[1]:.0f})<extra></extra>",
            )
        )
    fig.update_yaxes(title="£/MWh")
    fig.update_layout(hovermode="x unified")
    return style(fig, dark, height=400)
