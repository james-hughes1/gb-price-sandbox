import streamlit as st

from gb_price_sandbox import app_data
from gb_price_sandbox.models import MODELS
from gb_price_sandbox.viz import forecast as fv

st.set_page_config(page_title="Forecast sandbox", layout="wide")
dark = st.context.theme.type == "dark"
colors = app_data.model_colors(dark)
dashes = app_data.model_dashes()

st.title("Forecast sandbox")
st.markdown(
    "Day-ahead price forecasts, backtested *walk-forward*: each forecast for delivery day D is "
    "made on the morning of D-1, using only information published by then. Models are retrained "
    "every week on all the history available at the time."
)


@st.cache_data(show_spinner="Running walk-forward backtests (first run takes about a minute)…")
def backtests(test_start, test_end, names):
    results = app_data.run_backtests(test_start, test_end, list(names))
    board, by_slot = app_data.forecast_scores(results)
    return results, board, by_slot


test_end = app_data.default_test_end()
c1, c2 = st.columns([2, 1])
test_start, test_end = c1.slider(
    "Test period (delivery days)",
    min_value=app_data.DEFAULT_TEST_START,
    max_value=test_end,
    value=(app_data.DEFAULT_TEST_START, test_end),
    format="MMM YYYY",
)
names = c2.multiselect(
    "Models",
    list(MODELS),
    default=app_data.default_model_selection(app_data.DEFAULT_TEST_START, test_end),
    help="Models not yet backtested for this period can take several minutes on first run.",
)
if not names:
    st.info("Pick at least one model.")
    st.stop()

results, board, by_slot = backtests(test_start, test_end, tuple(names))

st.subheader("Leaderboard")
st.plotly_chart(fv.fig_metric_bars(board, "MAE", colors, dark), width="stretch", theme=None)
st.dataframe(
    board.style.format(
        {
            "MAE": "£{:.2f}",
            "Pinball": "£{:.2f}",
            "CRPS": "£{:.2f}",
            "Coverage 80%": "{:.0%}",
            "Periods": "{:,}",
        }
    ),
    width="stretch",
)
st.markdown(
    """
- **MAE**: mean absolute error of the central (median) forecast, in £/MWh.
- **Pinball loss**: the standard score for a quantile forecast. It penalises being on the wrong
  side of each quantile, weighted by how far out the quantile is.
- **CRPS**: scores the whole forecast distribution. For a point forecast it equals MAE, so the
  quantile models to come only beat these baselines on CRPS if their uncertainty bands are useful.
- **Coverage 80%**: the share of outcomes inside the 10–90% band. A well-calibrated model scores
  about 80%. Point forecasts score about 0%, because they have no band.

All models are scored on exactly the same half-hours.
"""
)

st.subheader("Forecasts against outcomes")
first = next(iter(results.values()))
days = sorted(first["delivery_day"].unique())
default_day = days[-7] if len(days) >= 7 else days[0]
c1, c2 = st.columns([2, 1])
start_day = c1.date_input("Week starting", value=default_day, min_value=days[0], max_value=days[-1])
span = c2.select_slider("Days shown", options=[1, 3, 7, 14], value=7)
st.plotly_chart(
    fv.fig_forecast_window(results, colors, start_day, span, dark, dashes),
    width="stretch",
    theme=None,
)

st.subheader("When are forecasts worst?")
st.plotly_chart(
    fv.fig_error_by_time_of_day(by_slot, colors, dark, dashes), width="stretch", theme=None
)
st.markdown(
    """
Errors are largest around midday, not at the evening peak. The likely reason is solar: midday
prices swing with how sunny it turns out to be, and no model has a solar forecast as an input yet.
That makes a solar forecast (PV_Live or Open-Meteo) the obvious next feature to add.
"""
)

st.subheader("The models")
st.markdown(
    """
**Seasonal naive.** The price in the same half-hour one week earlier. It captures the daily and
weekly rhythm and nothing else. Any useful model has to beat it.

**LEAR** (LASSO-estimated autoregressive model) is a standard benchmark in the price-forecasting
literature (Lago et al., 2021). For each half-hour of the day it fits a sparse linear model of:

- recent prices: 2, 3 and 7 days earlier, and the average of D-2;
- NESO's day-ahead demand and wind forecasts;
- the day of the week.

Prices go through an `asinh` transform first, which tames spikes and copes with negative
prices. LASSO switches off unhelpful inputs automatically. Yesterday's prices are deliberately
*not* used, because most of D-1 has not happened yet when the forecast is made.

**Avoiding look-ahead.** NESO revises its forecasts many times. Each input is the latest version
published before 10:00 on D-1, never a later revision. Tests in the repository check this, and
check that no model is ever trained on data it couldn't have had.
"""
)
