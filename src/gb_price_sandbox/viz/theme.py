"""Chart styling shared by every page: palette, fixed model colours, Plotly layout.

Colours are from the validated reference palette (categorical slots in fixed order,
checked for colour-vision-deficiency separation in both modes). Each mode has its own
steps rather than an automatic inversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class Palette:
    """Colour tokens for one mode."""

    series: tuple[str, ...]
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
)

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

ACTUAL = "Actual"


def palette(dark: bool) -> Palette:
    """Tokens for the requested mode."""
    return DARK if dark else LIGHT


def entity_colors(names: list[str], dark: bool, *, actual_names: tuple[str, ...] = ()) -> dict:
    """Fixed colour per entity: names in ``actual_names`` take ink, the rest take slots in order.

    Pass the *full* registry of names (not just the ones currently shown) so a model keeps
    its colour when others are filtered out.
    """
    pal = palette(dark)
    colors, slot = {}, 0
    for name in names:
        if name in actual_names:
            colors[name] = pal.ink_secondary
        else:
            colors[name] = pal.series[slot % len(pal.series)]
            slot += 1
    return colors


def style(
    fig: go.Figure,
    dark: bool,
    *,
    height: int = 380,
    legend: bool = True,
    grid: str = "y",
) -> go.Figure:
    """Apply the shared layout: transparent surface, hairline value grid, system font, top legend.

    Args:
        fig: Figure to style in place (also returned).
        dark: Dark-mode tokens.
        height: Figure height in px.
        legend: Show the legend (top left, horizontal).
        grid: Which axis carries gridlines: ``"y"`` (value axis of vertical charts),
            ``"x"`` (horizontal bars), ``"both"`` (scatter) or ``"none"``.
    """
    pal = palette(dark)
    axis = {
        "gridcolor": pal.grid,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": pal.baseline,
        "showline": False,
        "ticks": "",
        "automargin": True,
        "tickfont": {"color": pal.muted, "size": 12},
        "title": {"font": {"color": pal.ink_secondary, "size": 13}, "standoff": 10},
    }
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": pal.ink, "size": 13},
        margin={"l": 4, "r": 16, "t": 48 if legend else 16, "b": 4, "pad": 4},
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": pal.ink_secondary},
            "bgcolor": "rgba(0,0,0,0)",
            "itemwidth": 40,
        },
        hoverlabel={
            "font": {"family": FONT, "color": pal.ink},
            "bgcolor": pal.surface,
            "bordercolor": pal.baseline,
        },
    )
    fig.update_xaxes(**axis, showgrid=grid in ("x", "both"))
    fig.update_yaxes(**axis, showgrid=grid in ("y", "both"))
    # The axis opposite the grid gets a baseline instead.
    if grid == "y":
        fig.update_xaxes(showline=True)
    elif grid == "x":
        fig.update_yaxes(showline=True)
    fig.update_traces(marker_cornerradius=4, selector={"type": "bar"})
    return fig
