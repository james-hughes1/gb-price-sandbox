import streamlit as st

from gb_price_sandbox import app_data
from gb_price_sandbox.features.explore import PERIOD_LABELS, STATUS_LABELS
from gb_price_sandbox.viz import features as fx

st.set_page_config(page_title="Features", layout="wide")
dark = st.context.theme.type == "dark"


@st.cache_data(ttl="6h", show_spinner="Loading prices, forecasts, weather and solar data…")
def load():
    return app_data.load_feature_data()


@st.cache_data(ttl="6h", show_spinner=False)
def view(key, years, periods):
    frame, _ = load()
    return app_data.feature_view(app_data.filter_feature_frame(frame, years, periods), key)


@st.cache_data(ttl="6h", show_spinner="Computing correlations…")
def correlations(method, include_transforms, years, periods):
    frame, _ = load()
    filtered = app_data.filter_feature_frame(frame, years, periods)
    return app_data.feature_correlations(filtered, method, include_transforms)


frame, unavailable = load()
first_year, last_year = app_data.feature_filter_bounds(frame)

st.title("Features: what the models can see")
st.markdown(
    "A price forecast is only as good as its inputs. Pick a feature to see its raw series "
    "next to the transformed versions a model would use, how each relates to price, and "
    "**when it is known**: a feature is only allowed if it is available at 10:00 on D-1, when "
    "the forecast is made."
)
if unavailable:
    st.warning(f"Couldn't load: {', '.join(unavailable)}. Those features are hidden for now.")

# --- controls --------------------------------------------------------------------------------
options = app_data.feature_selector()
labels = dict(options)
c1, c2, c3 = st.columns([2, 1, 1])
keys = [k for k, _ in options]
# The selection lives in the URL (?feature=solar), so a view can be linked to directly.
requested = st.query_params.get("feature")
key = c1.selectbox(
    "Feature",
    keys,
    index=keys.index(requested) if requested in keys else 0,
    format_func=labels.get,
)
st.query_params["feature"] = key
years = c2.slider("Years", first_year, last_year, (first_year, last_year))
periods = c3.select_slider(
    "Time of day",
    options=list(range(48)),
    value=(0, 47),
    format_func=lambda p: PERIOD_LABELS[p],
)
years, periods = tuple(years), tuple(periods)

v = view(key, years, periods)
spec = v.spec

st.subheader(spec.title)
st.markdown(spec.summary)
used_by = f" · Used by {', '.join(spec.models)}" if spec.models else ""
st.caption(f"**{STATUS_LABELS[spec.status]}**{used_by} · Source: {spec.source}")

continuous = [x for x in v.variants if x.kind == "continuous"]
statuses = {spec.variant_status(x) for x in v.variants}


def corr_label(variant):
    """The variant's name with its correlation, for the details selector."""
    if variant.column not in v.correlations:
        return variant.label
    return f"{variant.label} (ρ {v.correlations[variant.column][0]:+.2f})"


def heading(variant):
    """Variant heading, tagged with its status on pages that mix used and candidate series."""
    tags = []
    if len(statuses) > 1:
        tags.append(STATUS_LABELS[spec.variant_status(variant)])
    if not variant.known_at_issue:
        tags.append("*not known at issue time: reference only*")
    return f"##### {variant.label}" + "".join(f" · {t}" for t in tags)


# --- correlation with each lag ---------------------------------------------------------------
if v.lag_profile is not None:
    st.markdown("##### How much does each past day tell you?")
    st.plotly_chart(fx.fig_lag_profile(v.lag_profile, dark), width="stretch", theme=None)

# --- headline correlations (pages with a handful of series) ----------------------------------
if v.correlations and len(v.correlations) <= 4:
    cols = st.columns(len(v.correlations))
    for col, variant in zip(
        cols, [c for c in v.variants if c.column in v.correlations], strict=True
    ):
        spearman, pearson = v.correlations[variant.column]
        col.metric(
            variant.short,
            f"{spearman:+.2f}",
            help=f"Correlation with {'asinh ' if variant.target == 'price_asinh' else ''}price. "
            f"Spearman (rank) {spearman:+.2f}; Pearson (straight-line) {pearson:+.2f}.",
        )
        col.caption(f"Spearman ρ with price · Pearson r {pearson:+.2f}")

# Pages with many series show one at a time, picked here (correlations in the labels).
if len(continuous) > 3:
    detail = st.selectbox("Show details for", continuous, format_func=corr_label)
    continuous = [detail]

# --- charts ----------------------------------------------------------------------------------
if continuous and spec.timeseries:
    st.markdown("##### Over time")
    st.caption("Faint line: daily mean. Bold line: 30-day average.")
    st.plotly_chart(
        fx.fig_variant_timeseries(v.daily, continuous, dark), width="stretch", theme=None
    )

# Variants with a price relationship get a row each (distribution | relationship). The rest
# (the target itself) show their distributions side by side.
paired = [x for x in continuous if x.column in v.relationships]
alone = [x for x in continuous if x.column not in v.relationships]
if alone:
    st.markdown("##### Distributions")
    for col, variant in zip(st.columns(len(alone)), alone, strict=True):
        table, outside, zeros = v.histograms[variant.column]
        col.plotly_chart(
            fx.fig_distribution(table, outside, zeros, variant, dark), width="stretch", theme=None
        )

for variant in paired:
    st.markdown(heading(variant))
    left, right = st.columns(2)
    table, outside, zeros = v.histograms[variant.column]
    left.plotly_chart(
        fx.fig_distribution(table, outside, zeros, variant, dark), width="stretch", theme=None
    )
    right.plotly_chart(
        fx.fig_relationship(v.relationships[variant.column], variant, dark),
        width="stretch",
        theme=None,
    )

for variant in (x for x in v.variants if x.kind == "categorical"):
    st.markdown(heading(variant))
    st.plotly_chart(
        fx.fig_by_category(v.categories[variant.column], variant, dark),
        width="stretch",
        theme=None,
    )

st.markdown(spec.explanation)
st.markdown(spec.source_note())
st.markdown(f"**When is it known?** {spec.timing}")

# --- correlation matrix ----------------------------------------------------------------------
st.divider()
st.subheader("How the features relate to each other")
c1, c2 = st.columns([1, 1])
method = c1.radio(
    "Correlation",
    ["spearman", "pearson"],
    format_func=lambda m: "Spearman (rank)" if m == "spearman" else "Pearson (straight-line)",
    horizontal=True,
)
include_transforms = c2.toggle("Include transformed versions", value=False)
matrix, matrix_labels, owner = correlations(method, include_transforms, years, periods)
highlight = {label for label, k in owner.items() if k == key}
st.plotly_chart(
    fx.fig_correlation(matrix, matrix_labels, dark, highlight), width="stretch", theme=None
)
st.caption(
    "Bold labels belong to the selected feature. The year and time-of-day filters above apply "
    "here too."
)
st.markdown(
    """
Each cell is the correlation between two features across every half-hour, from -1 (red: one
rises as the other falls) through 0 (grey: no relationship) to +1 (blue: they rise together).
The first row and column are the price itself.

**Reading it.**

- **The price lags form a block of strong positive correlations.** They carry largely the same
  information, the recent price level. This is *collinearity*. A plain linear regression would
  give them unstable, offsetting coefficients. LASSO copes by keeping one and shrinking the
  others towards zero. Tree models are unaffected but split the credit between them.
- **Demand, wind and residual demand are related by construction**, since residual demand is
  demand minus wind.
- **A low correlation with price doesn't make a feature useless.** Correlation only measures a
  relationship that holds across *all* half-hours. Solar is the clearest case: it only matters in
  daylight. Holidays matter on a few days a year. Narrow the time of day or the years to see
  these effects.

**Spearman vs Pearson.** Pearson measures how well a straight line fits, so a few extreme
prices can dominate it. Spearman uses ranks instead, so it is robust to spikes and isn't changed
by any order-preserving transform. That is why the asinh versions match their raw versions
exactly on Spearman. Switch to Pearson to see where the transform changes the straight-line
fit, which is what a linear model like LEAR relies on.
"""
)
