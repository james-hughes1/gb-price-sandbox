# GB Power Price Sandbox

## What this is
A multi-tab web app for exploring GB wholesale power prices, backtesting probabilistic price forecasts, and optimising battery dispatch against them.
Guiding question: **how much battery revenue does forecast error cost, and do quantile forecasts win some of it back?**
Headline metric: **capture rate** = revenue from forecast-driven dispatch / revenue with perfect foresight.

The tabs should ideally be visually intuitive, interactive where this enriches the user understanding, and have explanations below the visuals explaining relevant history, regulation, or policy regarding the GB power market. Ideally it should be readable both on a laptop or a phone, but laptop takes priority.

## Tabs
1. **Background:** historical GB prices, demand, wind/solar and gas, plus a plain-English explanation of how prices are set (merit order, day-ahead vs balancing).
2. **Forecast sandbox:** run day-ahead price models with walk-forward backtests and compare them on a leaderboard.
3. **Battery optimiser:** LP/MILP dispatch of a battery; perfect foresight vs dispatch on each model's forecast.
4. **Scenarios (later, illustrative only):** gas price shocks and weather-year resampling feeding through to price and revenue.

## Data
- Elexon Insights API: Market Index Price (MID, half-hourly) as the day-ahead price proxy, plus system price.
- Elexon Insights API: NESO's demand (NDF, half-hourly) and wind (WINDFOR, hourly) **forecasts** (not actuals), with every published vintage time-stamped. Not the NESO portal's "historic" files: they keep only the final revision, often published after delivery.
- **Study period starts 2022-01-01** (earliest NDF vintage history on Elexon).
- Open-Meteo (historical forecasts), PV_Live, a gas price series, a GB calendar.
- Cache raw pulls locally (parquet); loaders must be idempotent.

## Modelling
- Baselines: seasonal naive (same period last week); LEAR (LASSO autoregressive model with exogenous inputs)
- Models: linear quantile regression, LightGBM quantile (q = 0.1, 0.5, 0.9); optionally conformal intervals.
- Walk-forward validation: expanding window, forecast issued D-1 for delivery day D, weekly retrain.
- Metrics: MAE, pinball loss, interval coverage, CRPS (approx. from quantiles).
- **No leakage:** features must be available at forecast issue time. Test this.

## Optimisation
- PuLP with the HiGHS solver.
- Battery: power (MW), energy (MWh), round-trip efficiency, SoC bounds, cycle limit.
- MILP binary to prevent simultaneous charge/discharge (matters at negative prices).
- Report revenue, cycles and capture rate per forecast model.

## Stack
Python 3.12, uv, pandas, numpy, scikit-learn, statsmodels, LightGBM, PuLP/HiGHS, Plotly, Streamlit (multi-page app).

## Engineering practices
- `src/` layout package; the app only calls the package, no logic in UI files.
- Type hints and docstrings on public functions.
- pytest: unit tests for loaders, feature timing (no leakage), metrics, and the LP (check against a small hand-solved example).
- Ruff (lint + format), pre-commit hooks.
- GitHub Actions CI: lint + tests on every push/PR.
- Dockerfile for the app; config via environment variables, no secrets in the repo.
- **Never commit.** Claude must not run `git commit`, `push`, `merge`, `rebase`, `cherry-pick`, `revert` or `am` in this repo, and should not offer to. James reviews and commits every change himself. Leave all work uncommitted in the working tree. (Enforced by deny rules in `.claude/settings.json`.).
- README: the question, results, how to run, data sources and limitations.
