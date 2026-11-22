import streamlit as st

from gb_price_sandbox import app_data
from gb_price_sandbox.optimise import PERFECT_FORESIGHT
from gb_price_sandbox.viz import battery as bv

st.set_page_config(page_title="Battery optimiser", layout="wide")
dark = st.context.theme.type == "dark"
colors = app_data.model_colors(dark)

st.title("Battery optimiser")
st.markdown(
    "Each day the battery's schedule is fixed from a **forecast** of tomorrow's prices, then paid "
    "the **actual** prices. *Perfect foresight* plans against the actual prices: the most any "
    "schedule could have earned. **Capture rate** is forecast-driven revenue as a share of that "
    "ceiling."
)

try:
    pretrained = app_data.battery_inputs()
except app_data.PretrainedMissingError:
    st.error(
        "The forecasts this page uses haven't been generated yet. Run "
        "`uv run python -m gb_price_sandbox.pretrain` (under a minute), then reload."
    )
    st.stop()

st.caption(
    f"Forecasts from **{' and '.join(pretrained.results)}**, pretrained walk-forward for "
    f"{pretrained.test_start:%b %Y} to {pretrained.test_end:%b %Y}. This page only optimises; "
    "to compare other models, use the Forecast sandbox."
)

# --- options (a form: nothing runs until Optimise) -------------------------------------------
with st.form("battery"):
    c1, c2, c3, c4 = st.columns(4)
    power = c1.slider("Power (MW)", 0.5, 10.0, 1.0, 0.5)
    duration = c2.select_slider(
        "Duration (hours)", options=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0], value=2.0
    )
    rte = c3.slider("Round-trip efficiency", 0.70, 0.95, 0.88, 0.01, format="%.2f")
    cycles = c4.slider("Max cycles per day", 0.5, 3.0, 1.0, 0.5)
    c1, c2 = st.columns([3, 1])
    window = c1.select_slider(
        "Period", options=list(app_data.BATTERY_WINDOWS), value="Last 12 months"
    )
    c2.write("")  # align the button with the slider
    run = c2.form_submit_button("Optimise", type="primary", width="stretch")

options = (power, duration, rte, cycles, window)
if run:
    bar = st.progress(0.0, text="Starting…")
    st.session_state["battery_result"] = (
        options,
        app_data.run_battery_page(
            *options, progress=lambda f, m: bar.progress(min(f, 1.0), text=m)
        ),
    )
    bar.empty()

stored = st.session_state.get("battery_result")
if stored is None:
    st.info("Set the battery above and press **Optimise**.")
    st.stop()
ran_with, result = stored
if ran_with != options:
    st.caption("Options changed since the last run: press **Optimise** to update.")

# --- results ---------------------------------------------------------------------------------
table = result.table
perfect = table.loc[PERFECT_FORESIGHT]
models = [n for n in table.index if n != PERFECT_FORESIGHT]
best = table.loc[models, "Capture rate"].idxmax()
b = result.battery

st.divider()
st.header("Results")
st.caption(
    f"{b.power_mw:g} MW / {b.energy_mwh:g} MWh, {b.round_trip_efficiency:.0%} round trip, "
    f"at most {b.max_cycles_per_day:g} cycle{'' if b.max_cycles_per_day == 1 else 's'} a day"
    " · delivery days "
    f"{result.window[0]:%d %b %Y} to {result.window[1]:%d %b %Y}"
)
c = st.columns(3)
c[0].metric("Perfect-foresight revenue", f"£{perfect['Revenue (GBP)']:,.0f}")
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
c1, c2 = st.columns([2, 1])
day = c1.select_slider(
    "Delivery day",
    options=result.days,
    value=result.days[-1],
    format_func=lambda d: f"{d:%a %d %b %Y}",
)
shown = c2.multiselect("Schedules", [PERFECT_FORESIGHT, *models], default=[PERFECT_FORESIGHT, best])
st.plotly_chart(
    bv.fig_dispatch_day(
        result.schedules, result.forecasts, shown, colors, day, dark, app_data.model_dashes()
    ),
    width="stretch",
    theme=None,
)
st.markdown(
    "The top panel shows the actual price and what each model expected; the bottom, how full "
    "the battery is. Each schedule is planned from its own price line, so when a forecast puts "
    "the peak in the wrong half-hour, the battery discharges at the wrong time."
)

# --- the optimisation ------------------------------------------------------------------------
st.divider()
st.subheader("The optimisation")
st.markdown(
    r"""
Each day is planned once, from a price curve $\lambda_t$: the forecast, or the actual prices
for perfect foresight. The plan is the solution of a small linear program.

**Variables**, for each half-hour $t = 1, \dots, T$ of the day ($T = 48$, each $\Delta = 0.5$ h):
$c_t, d_t \ge 0$, the charging and discharging power at the grid connection (MW), and $e_t$,
the energy stored at the end of the half-hour (MWh).

**Parameters**: power rating $P$ (MW); storage limits $\underline{e} \le \overline{e}$ (MWh),
here $0$ and the capacity $E$; starting level $e_0 = E/2$; one-way efficiency
$\eta = \sqrt{\text{round-trip efficiency}}$; at most $N$ full cycles per day.

**The LP**
$$
\begin{aligned}
\max_{c,\,d,\,e}\quad & \sum_{t=1}^{T} \lambda_t\,(d_t - c_t)\,\Delta
  && \text{revenue at the planning prices} \\
\text{s.t.}\quad & e_t = e_{t-1} + \Big(\eta\,c_t - \tfrac{d_t}{\eta}\Big)\Delta
  && \text{energy balance, with losses} \\
& \underline{e} \le e_t \le \overline{e}, \qquad 0 \le c_t,\, d_t \le P
  && \text{storage and power limits} \\
& e_T \ge e_0 && \text{end the day at least as full} \\
& \textstyle\sum_t d_t\,\Delta \le N\,(\overline{e} - \underline{e})
  && \text{cycle cap (a proxy for wear)}
\end{aligned}
$$

**The MILP.** At negative prices the LP has a loophole: charge and discharge *at the same time*,
importing more energy than the battery keeps and losing the excess in conversion, to get paid
for consuming it. A real battery can't do both at once. One binary per half-hour,
$u_t \in \{0, 1\}$ (1 = charging), closes the loophole:

$$
c_t \le P\,u_t, \qquad d_t \le P\,(1 - u_t).
$$

That turns the LP into a **mixed-integer** linear program. HiGHS solves it by branch and bound
(solving LP relaxations and branching on fractional $u_t$) in milliseconds per day.

**Scoring.** A schedule planned on the forecast is paid the actual prices $p_t$:
$R = \sum_t p_t\,(d_t - c_t)\,\Delta$. Perfect foresight plans with $\lambda_t = p_t$, giving the
ceiling $R^\star$, and the **capture rate** is $R / R^\star$.

*Limitations:* wholesale arbitrage only, settled at MID. No balancing or ancillary revenue, no
degradation cost beyond the cycle cap, and no re-planning during the day.
"""
)
