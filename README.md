# GB Power Price Sandbox

**How much battery revenue does forecast error cost, and do quantile forecasts win some of it back?**

A multi-tab Streamlit app for exploring GB wholesale power prices, backtesting probabilistic
day-ahead price forecasts, and optimising battery dispatch against them.

Headline metric: **capture rate** = revenue from forecast-driven dispatch ÷ revenue with perfect
foresight.

## Results

_To come._

## How to run

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is fetched automatically if missing).

```bash
uv sync                                  # create .venv and install deps
uv run pre-commit install                # enable git hooks
uv run streamlit run app/Home.py         # launch the app
uv run pytest                            # tests (offline)
uv run pytest -m network                 # live API smoke tests
uv run ruff check . && uv run ruff format --check .
```

Configuration is via environment variables (prefix `GBPS_`); see `.env.example`.

### Docker

```bash
docker build -t gb-price-sandbox .
docker run --rm -p 8501:8501 -v "$PWD/data:/app/data" gb-price-sandbox
```

## Project layout

```
src/gb_price_sandbox/
  config.py        settings from env vars
  data/            API loaders + idempotent parquet cache
  features/        feature engineering (issue-time safe)
  models/          seasonal naive, LEAR, quantile regression, LightGBM quantile
  backtest/        walk-forward validation
  metrics.py       MAE, pinball, coverage, CRPS
  optimise/        battery MILP (PuLP + HiGHS)
app/               Streamlit pages (UI only)
tests/
```

## Data sources

- **Elexon Insights API:** Market Index Price (half-hourly, day-ahead proxy) and system price.
- **Elexon Insights API (NESO forecasts):** national demand forecast (NDF, half-hourly) and wind
  generation forecast (WINDFOR, hourly), with every published vintage time-stamped, so features
  are built only from forecasts available at the D-1 issue time. Coverage starts in 2022, which sets
  the study period. (The NESO portal's "historic" files are unsuitable: they keep one revision per
  period, often published after delivery.)
- **Open-Meteo** historical forecasts, **PV_Live** (Sheffield Solar), a gas price series, a GB
  calendar.

## Limitations

_To come._
