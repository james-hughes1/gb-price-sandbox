"""Figures for the Features tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gb_price_sandbox.features.explore import Variant
from gb_price_sandbox.viz.theme import palette, style

#: Diverging pair from the reference palette: red (negative) and blue (positive) with a
#: neutral grey midpoint, stepped separately for each mode.
_DIVERGING = {
    False: ("#e34948", "#f0efec", "#2a78d6"),
    True: ("#e66767", "#383835", "#3987e5"),
}


def _alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _axis_title(v: Variant) -> str:
    return f"{v.label} ({v.unit})" if v.unit else v.label


def _target_title(target: str) -> str:
    return "Price (asinh scale)" if target == "price_asinh" else "Price (£/MWh)"


def _fmt(unit: str) -> str:
    return ".2f" if unit in ("", "share", "std devs", "asinh scale") else ",.1f"


def fig_variant_timeseries(daily: pd.DataFrame, variants: list[Variant], dark: bool) -> go.Figure:
    """Daily mean of each variant (faint) under its 30-day average (bold), one panel each.

    Variants have different units, so each gets its own panel and y-axis rather than sharing
    one (no dual axes). Panels share the time axis.
    """
    pal = palette(dark)
    color = pal.series[0]
    n = len(variants)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08 if n > 1 else 0.0,
        subplot_titles=[_axis_title(v) for v in variants],
    )
    for i, v in enumerate(variants, start=1):
        series = daily[v.column]
        rolling = series.rolling(30, min_periods=15, center=True).mean()
        fmt = _fmt(v.unit)
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=series,
                mode="lines",
                line={"width": 1, "color": _alpha(color, 0.35)},
                name="Daily mean",
                showlegend=i == 1,
                legendgroup="daily",
                hovertemplate=f"%{{x|%a %d %b %Y}}: %{{y:{fmt}}}<extra>{v.short}</extra>",
            ),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=rolling,
                mode="lines",
                line={"width": 2, "color": color},
                name="30-day average",
                showlegend=i == 1,
                legendgroup="rolling",
                hovertemplate=f"30-day average %{{y:{fmt}}}<extra>{v.short}</extra>",
            ),
            row=i,
            col=1,
        )
    fig.update_annotations(font={"color": pal.ink_secondary, "size": 13}, x=0, xanchor="left")
    fig.update_layout(hovermode="x unified")
    # One series per panel, so no legend: the page caption explains faint vs bold.
    style(fig, dark, height=100 + 190 * n, legend=False)
    fig.update_xaxes(showline=True)
    fig.update_layout(margin={"t": 32})
    return fig


def _panel_title(fig: go.Figure, text: str, dark: bool, note: str = "") -> None:
    """Small title above the plot, with an optional muted second line (kept short for phones)."""
    pal = palette(dark)
    if note:
        text += f"<br><span style='font-size:11px;color:{pal.muted}'>{note}</span>"
    fig.update_layout(
        title={
            "text": text,
            "font": {"size": 13, "color": pal.ink_secondary},
            "x": 0,
            "xref": "paper",
            "y": 1,
            "yref": "paper",
            "yanchor": "bottom",
            "pad": {"b": 14},
        },
        margin={"t": 72},
        legend={"y": 1.0, "yanchor": "bottom"},
    )


def fig_distribution(
    table: pd.DataFrame, outside: float, zeros: float, variant: Variant, dark: bool
) -> go.Figure:
    """Histogram (share of half-hours per bin) over the central 99% of values.

    If many values are exactly zero (solar at night), they are left out of the bars and their
    share is given in the title instead.
    """
    pal = palette(dark)
    fmt = _fmt(variant.unit)
    fig = go.Figure(
        go.Bar(
            x=table["mid"],
            y=table["share"],
            width=(table["right"] - table["left"]).to_numpy(),
            marker={"color": pal.series[0], "line": {"width": 1, "color": pal.surface}},
            customdata=table[["left", "right"]].to_numpy(),
            hovertemplate=f"%{{customdata[0]:{fmt}}} to %{{customdata[1]:{fmt}}}"
            "<br>%{y:.1%} of half-hours<extra></extra>",
        )
    )
    notes = []
    if zeros:
        notes.append(f"{zeros:.0%} exactly zero (not shown)")
    if outside > 0.0005:
        notes.append(f"{outside:.1%} off-scale")
    fig.update_xaxes(title=_axis_title(variant))
    fig.update_yaxes(title="Share of half-hours", tickformat=".0%")
    fig.update_layout(bargap=0)
    fig = style(fig, dark, height=320, legend=False)
    _panel_title(fig, "Distribution", dark, " · ".join(notes))
    fig.update_traces(marker_cornerradius=0)
    return fig


def fig_relationship(binned: pd.DataFrame, variant: Variant, dark: bool) -> go.Figure:
    """Median price (line) and 10th to 90th percentile band within 20 equal-count bins."""
    pal = palette(dark)
    color = pal.series[0]
    fmt = _fmt(variant.unit)
    yfmt = ".2f" if variant.target == "price_asinh" else ",.0f"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.concat([binned["x"], binned["x"][::-1]]),
            y=pd.concat([binned["p90"], binned["p10"][::-1]]),
            fill="toself",
            fillcolor=_alpha(color, 0.18),
            line={"width": 0},
            name="10th to 90th percentile",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=binned["x"],
            y=binned["p50"],
            mode="lines+markers",
            line={"width": 2, "color": color},
            marker={"size": 8, "color": color, "line": {"width": 2, "color": pal.surface}},
            name="Median",
            customdata=binned[["p10", "p90", "n"]].to_numpy(),
            hovertemplate=f"{variant.short} ≈ %{{x:{fmt}}}<br>Median price %{{y:{yfmt}}}"
            f"<br>10th to 90th: %{{customdata[0]:{yfmt}}} to %{{customdata[1]:{yfmt}}}"
            "<br>%{customdata[2]:,} half-hours<extra></extra>",
        )
    )
    fig.update_xaxes(title=_axis_title(variant))
    fig.update_yaxes(title=_target_title(variant.target))
    fig = style(fig, dark, height=320, grid="both")
    _panel_title(fig, "Relationship with price", dark, "Median, with 10th to 90th percentile band")
    fig.update_layout(showlegend=False)
    return fig


def fig_by_category(table: pd.DataFrame, variant: Variant, dark: bool) -> go.Figure:
    """Price by category: bars of the mean (few categories) or a line with a band (many)."""
    pal = palette(dark)
    color = pal.series[0]
    yfmt = ".2f" if variant.target == "price_asinh" else ",.0f"
    custom = table[["p10", "p50", "p90", "n"]].to_numpy()
    hover = (
        f"%{{x}}<br>Mean price %{{y:{yfmt}}}<br>Median %{{customdata[1]:{yfmt}}}"
        f"<br>10th to 90th: %{{customdata[0]:{yfmt}}} to %{{customdata[2]:{yfmt}}}"
        "<br>%{customdata[3]:,} half-hours<extra></extra>"
    )
    fig = go.Figure()
    if len(table) > 12:
        x = list(table.index)
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(table["p90"]) + list(table["p10"])[::-1],
                fill="toself",
                fillcolor=_alpha(color, 0.18),
                line={"width": 0},
                name="10th to 90th percentile",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=table["mean"],
                mode="lines",
                line={"width": 2, "color": color},
                name="Mean",
                customdata=custom,
                hovertemplate=hover,
            )
        )
        fig.update_xaxes(tickvals=x[::8])
        fig.update_layout(hovermode="x")
        legend = True
    else:
        fig.add_trace(
            go.Bar(
                x=list(table.index),
                y=table["mean"],
                marker={"color": color},
                marker_line_width=0,
                customdata=custom,
                hovertemplate=hover,
            )
        )
        fig.update_layout(bargap=0.35)
        legend = False
    fig.update_xaxes(title=variant.label)
    fig.update_yaxes(title="Mean " + _target_title(variant.target).lower())
    return style(fig, dark, height=340, legend=legend)


def fig_lag_profile(profile: pd.DataFrame, dark: bool) -> go.Figure:
    """Correlation of price with itself k days earlier, one bar per lag.

    Bars are coloured by whether a model uses the lag; the legend names both groups, so the
    distinction isn't carried by colour alone.
    """
    pal = palette(dark)
    fig = go.Figure()
    for used, name, color in (
        (True, "Used by the models", pal.series[0]),
        (False, "Not used", pal.muted),
    ):
        sub = profile[profile["used"] == used]
        fig.add_trace(
            go.Bar(
                x=sub["lag_days"],
                y=sub["correlation"],
                name=name,
                marker={"color": color},
                marker_line_width=0,
                hovertemplate="%{x} days earlier: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Lag (days)", tickvals=list(profile["lag_days"]), range=[0.4, 14.6])
    fig.update_yaxes(title="Spearman correlation with price", rangemode="tozero")
    fig.add_annotation(
        x=1,
        y=0,
        yanchor="bottom",
        text="1 day:<br>not yet<br>known",
        showarrow=False,
        font={"color": pal.muted, "size": 11},
    )
    fig.update_layout(bargap=0.3, barmode="overlay")
    return style(fig, dark, height=320)


def fig_correlation(
    matrix: pd.DataFrame, labels: list[str], dark: bool, highlight: set[str] | None = None
) -> go.Figure:
    """Correlation heatmap on a diverging scale fixed to [-1, 1], neutral at zero.

    Labels in ``highlight`` (the selected feature's variants) are set in bold.
    """
    pal = palette(dark)
    neg, mid, pos = _DIVERGING[dark]
    highlight = highlight or set()
    shown = [f"<b>{lab}</b>" if lab in highlight else lab for lab in labels]
    n = len(labels)
    show_text = n <= 18
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=shown,
            y=shown,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[[0, neg], [0.5, mid], [1, pos]],
            xgap=2,
            ygap=2,
            text=matrix.round(2).to_numpy(),
            texttemplate="%{text:.2f}" if show_text else None,
            textfont={"size": 10, "color": pal.ink},
            colorbar={
                "thickness": 10,
                "outlinewidth": 0,
                "tickfont": {"color": pal.muted, "size": 11},
                "tickvals": [-1, -0.5, 0, 0.5, 1],
            },
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45)
    fig = style(fig, dark, height=max(420, 28 * n + 160), legend=False, grid="none")
    fig.update_xaxes(showline=False)
    fig.update_layout(plot_bgcolor=pal.surface)
    return fig
