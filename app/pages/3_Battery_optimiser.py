import streamlit as st

from gb_price_sandbox import app_data
from gb_price_sandbox.models import MODELS
from gb_price_sandbox.optimise import PERFECT_FORESIGHT
from gb_price_sandbox.viz import battery as bv

st.set_page_config(page_title="Battery optimiser", layout="wide")
dark = st.context.theme.type == "dark"
colors = app_data.model_colors(dark)

st.title("Battery optimiser")
st.markdown(
    "Each day the battery's schedule is fixed from a **forecast** of tomorrow's prices, then paid "
    "the **actual** prices. *Perfect foresight* optimises against the actual prices: the most "
    "that schedule could have earned. **Capture rate** is forecast-driven revenue as a share of "
    "that ceiling."
)

with st.sidebar:
    st.header("Battery")
    power = st.slider("Power (MW)", 0.5, 10.0, 1.0, 0.5)
    duration = st.select_slider(
        "Duration (hours)", options=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0], value=2.0
    )
    rte = st.slider("Round-trip efficiency", 0.70, 0.95, 0.88, 0.01, format="%.2f")
    cycles = st.slider("Max cycles per day", 0.5, 3.0, 1.0, 0.5)


window = st.select_slider("Period", options=list(app_data.BATTERY_WINDOWS), value="Last 12 months")


@st.cache_data(show_spinner="Optimising dispatch for every day (one MILP per day per model)…")
def run(power, duration, rte, cycles, window):
    return app_data.run_battery_page(power, duration, rte, cycles, window)


table, schedules, forecasts, day_options = run(power, duration, rte, cycles, window)
perfect = table.loc[PERFECT_FORESIGHT]

c = st.columns(3)
c[0].metric("Perfect-foresight revenue", f"£{perfect['Revenue (GBP)']:,.0f}")
best = table.drop(PERFECT_FORESIGHT)["Capture rate"].idxmax()
c[1].metric(f"Best capture rate ({best})", f"{table.loc[best, 'Capture rate']:.0%}")
c[2].metric(
    "Revenue lost to forecast error",
    f"£{perfect['Revenue (GBP)'] - table.loc[best, 'Revenue (GBP)']:,.0f}",
    help=f"With the best model, over {int(perfect['Days'])} days",
)

st.subheader("Capture rate by forecast")
st.plotly_chart(bv.fig_capture_rates(table, colors, dark), width="stretch", theme=None)
st.dataframe(
    table.style.format(
        {"Revenue (GBP)": "£{:,.0f}", "Cycles": "{:,.0f}", "Capture rate": "{:.1%}", "Days": "{:,}"}
    ),
    width="stretch",
)

st.subheader("One day in detail")
day = st.select_slider(
    "Delivery day",
    options=day_options,
    value=day_options[-1],
    format_func=lambda d: f"{d:%a %d %b %Y}",
)
shown = st.multiselect(
    "Schedules", [PERFECT_FORESIGHT, *MODELS], default=[PERFECT_FORESIGHT, list(MODELS)[-1]]
)
st.plotly_chart(
    bv.fig_dispatch_day(schedules, forecasts, shown, colors, day, dark),
    width="stretch",
    theme=None,
)
st.markdown(
    """
The top panel shows the actual price and what each model expected. The bottom panel shows how
full the battery is. Each schedule is planned from its own price line, so when a forecast puts
the peak in the wrong half-hour, the battery discharges at the wrong time.
"""
)

st.subheader("How the optimiser works")
st.markdown(
    """
For each day a small **mixed-integer linear program** (PuLP + HiGHS) chooses how much to charge or
discharge in every half-hour to maximise revenue, subject to:

- power and energy limits, and round-trip losses (split evenly between charging and discharging);
- a cap on full cycles per day, a proxy for battery degradation;
- ending the day at least as full as it started, so days are independent;
- **a binary per half-hour that stops it charging and discharging at once.** At negative prices
  an LP without this would do both at the same time, wasting energy through losses to get paid
  for consuming it. That isn't physically possible.

Limitations: wholesale arbitrage only, settled at MID. There are no balancing or ancillary
revenues, no degradation cost beyond the cycle cap, and each day is planned once with no
intraday re-optimisation.
"""
)
