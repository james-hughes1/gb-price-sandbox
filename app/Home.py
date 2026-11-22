"""Streamlit entry point. UI only: all logic lives in the gb_price_sandbox package."""

import streamlit as st

st.set_page_config(page_title="GB Power Price Sandbox", page_icon="⚡", layout="wide")

st.title("GB Power Price Sandbox")
st.markdown(
    """
**How much battery revenue does forecast error cost, and do quantile forecasts win some of it
back?**

Headline metric: **capture rate** = revenue from forecast-driven dispatch ÷ revenue with
perfect foresight.

Use the sidebar to explore:

1. **Background**: historical GB prices, demand, renewables and gas, and how prices are set.
2. **Features**: every model input and candidate feature, its transformations, how it
   relates to price, and a correlation matrix.
3. **Forecast sandbox**: day-ahead price models backtested walk-forward, on a leaderboard.
4. **Battery optimiser**: perfect-foresight vs forecast-driven dispatch.
5. **Scenarios**: illustrative gas shocks and weather-year resampling.
"""
)
