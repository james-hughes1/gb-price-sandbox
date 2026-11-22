import streamlit as st

from gb_price_sandbox import app_data
from gb_price_sandbox.merit_order import build_stack, clear
from gb_price_sandbox.viz import background as bg

st.set_page_config(page_title="Background", layout="wide")
dark = st.context.theme.type == "dark"


@st.cache_data(ttl="6h", show_spinner="Loading GB price and forecast data…")
def load():
    data = app_data.load_study_data()
    return data, bg.daily_summary(data), bg.headline_stats(data)


data, daily, stats = load()
years = stats["yearly_mean"]

st.title("Background: GB wholesale power prices")
st.caption(
    f"Half-hourly Market Index Price (MID), {stats['first_day']:%d %b %Y} to "
    f"{stats['last_day']:%d %b %Y}. Demand and wind are NESO's day-ahead forecasts."
)

cols = st.columns(4)
first_year, last_year = min(years), max(years)
cols[0].metric(f"{first_year} average", f"£{years[first_year]:.0f}/MWh")
cols[1].metric(
    f"{last_year} average (to date)",
    f"£{years[last_year]:.0f}/MWh",
    # Sign first so Streamlit picks the arrow direction; grey because cheaper isn't "good" here.
    delta=f"{'-' if years[last_year] < years[first_year] else '+'}"
    f"£{abs(years[last_year] - years[first_year]):.0f} vs {first_year}",
    delta_color="off",
)
cols[2].metric("Highest half-hour", f"£{stats['max_period_price']:,.0f}")
cols[3].metric("Hours below £0", f"{stats['negative_hours']:,.0f}")

# --- price history ---------------------------------------------------------------------------
st.subheader("Prices since 2022")
st.plotly_chart(bg.fig_price_history(daily, dark), width="stretch", theme=None)
st.markdown(
    f"""
Prices in 2022 were extraordinary. Russia's invasion of Ukraine in February 2022 pushed European
gas prices to record highs, and GB power prices followed. The highest daily average in this data
was **£{stats["peak_day_mean"]:,.0f}/MWh on {stats["peak_day"]:%d %B %Y}**. Prices have since
fallen back but remain above pre-crisis levels, because gas still sets the price most of the time
(see the merit order below).

Hover over a day to see its range from the cheapest to the most expensive half-hour. **That
spread is what a battery earns from**: charge in the cheap half-hours, discharge in the dear ones.
"""
)

# --- merit order -----------------------------------------------------------------------------
st.subheader("How the price is set: the merit order")
st.markdown(
    "Drag the sliders. Generators offer power at roughly their running cost. The market "
    "takes the cheapest offers first until demand is met, and **every generator is paid the "
    "price of the last (most expensive) one needed**. This is *pay-as-clear*."
)
c1, c2, c3 = st.columns(3)
demand = c1.slider("Demand (GW)", 15, 45, 30)
wind = c2.slider("Wind output (GW)", 0, 22, 8)
gas = c3.slider("Gas price (p/therm)", 30, 400, 80, step=10)
stack = build_stack(wind, gas)
result = clear(stack, demand)
if result.price_gbp_mwh is None:
    st.error(
        "Demand exceeds the available supply in this simplified stack. In reality, imports, "
        "storage, demand response and very high scarcity prices fill the gap."
    )
else:
    st.metric(
        "Clearing price",
        f"£{result.price_gbp_mwh:,.0f}/MWh",
        help=f"Set by: {result.marginal_technology}",
    )
    st.caption(f"Marginal (price-setting) plant: **{result.marginal_technology}**")
st.plotly_chart(bg.fig_merit_order(stack, demand, result, dark), width="stretch", theme=None)
st.markdown(
    """
Things to try:

- **Raise the gas price.** When gas plant is marginal the power price moves almost one-for-one
  with fuel cost. That is why 2022 was so expensive.
- **Raise the wind.** Wind has almost no running cost, so it pushes the whole stack to the right
  and a cheaper plant ends up setting the price.
- **High wind with low demand.** Prices can go negative. Older wind farms on the Renewables
  Obligation, and early Contracts for Difference, are paid per MWh they generate, so they will pay
  to keep running. Newer CfD contracts don't pay out when prices are negative, so those farms
  tend to switch off instead.

This stack is stylised: round-number capacities and costs, with carbon at £60/t. Real offers
also reflect start-up costs, plant flexibility, interconnector flows, storage, and scarcity.
"""
)

st.subheader("The merit order in the data")
st.plotly_chart(bg.fig_price_vs_residual(daily, dark), width="stretch", theme=None)
st.markdown(
    """
Each dot is a day. *Residual demand* is demand minus wind: what the rest of the fleet (mostly
gas) has to supply. More residual demand means a more expensive plant at the margin. In 2022 the
whole curve sat far higher because gas was so expensive. That is why forecasts of demand and
wind are useful features for a price model.
"""
)

st.subheader("Demand and wind")
st.plotly_chart(bg.fig_demand_wind(daily, dark), width="stretch", theme=None)
st.markdown(
    """
Demand peaks in winter (heating and lighting) and is lowest on summer weekends. Wind is also
strongest in winter but varies far more from day to day. GB has one of the largest offshore wind
fleets in the world, so wind is the main swing factor in prices.
"""
)

# --- shape of the day ------------------------------------------------------------------------
st.subheader("The shape of a day")
st.plotly_chart(bg.fig_daily_shape(bg.daily_shape_by_year(data), dark), width="stretch", theme=None)
st.markdown(
    """
Average price by time of day. The evening peak (around 18:00) is when demand is highest and solar
has gone. Overnight used to be the cheapest time of day. As solar capacity has grown, the midday
dip has deepened, and in 2026 midday overtook overnight as the cheapest time. The gap between
trough and evening peak is the daily arbitrage a battery tries to capture.
"""
)

st.subheader("Negative prices")
st.plotly_chart(
    bg.fig_negative_hours(bg.negative_hours_by_month(data), dark), width="stretch", theme=None
)
st.markdown(
    """
Negative prices come with high wind and low demand. They are about three times as common at
weekends as on weekdays, and cluster around midday (when solar adds to the wind) and overnight.
For a battery they are the best time to charge, since it is *paid* to take power. They are also
why the optimiser needs a rule against charging and discharging at the same time: without it, a
model could "burn" energy through round-trip losses just to collect the negative price.
"""
)

# --- market structure ------------------------------------------------------------------------
st.subheader("How the GB market works")
st.markdown(
    """
**Wholesale trading.** Most power is traded ahead of time: months ahead on forward markets, then
day-ahead and intraday on power exchanges (EPEX SPOT and Nord Pool). Since leaving the EU's
single electricity market at the start of 2021, GB's day-ahead auctions run separately from the
coupled European auction. Cross-border capacity on interconnectors is sold separately rather
than allocated implicitly.

**The Market Index Price (MID)** used here is published by Elexon. It is the volume-weighted
price of short-term trades on the exchanges in each half-hour. It is a widely used reference for
the GB short-term price and stands in here for the day-ahead price.

**Balancing.** Each half-hour is a *settlement period*. One hour before it starts (*gate
closure*), parties lock in their positions. From then on NESO, the system operator, keeps supply
and demand matched in real time using the **Balancing Mechanism**, accepting offers to increase
or bids to reduce generation or demand. Anyone out of balance pays or receives the **imbalance
(system) price**. Since November 2015 that is a single price for both directions.

**Policy context.** Renewables are supported through Contracts for Difference (CfDs), which top
generators up to a fixed strike price, and the older Renewables Obligation. In 2025 the
government's Review of Electricity Market Arrangements (REMA) decided to keep a single national
wholesale price rather than split GB into price zones.

**What a battery earns.** Real batteries stack several revenue streams: wholesale arbitrage,
the Balancing Mechanism, frequency-response services and the Capacity Market. This app looks at
wholesale arbitrage only. It is the stream that depends most directly on forecasting prices well.
"""
)
