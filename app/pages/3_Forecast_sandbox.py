import streamlit as st

from gb_price_sandbox import app_data
from gb_price_sandbox.backtest import METRICS
from gb_price_sandbox.models.docs import FAMILY_DOCS
from gb_price_sandbox.viz import forecast as fv

st.set_page_config(page_title="Forecast sandbox", layout="wide")
dark = st.context.theme.type == "dark"
colors = app_data.model_colors(dark)
dashes = app_data.model_dashes()

st.title("Forecast sandbox")
st.markdown(
    "Set up an experiment, then press **Train**. Every model is backtested *walk-forward*: each "
    "forecast for delivery day D is made at 10:00 on D-1 from information published by then, "
    "and models are refitted as the holdout period rolls forward. Results are cached, so "
    "re-running a configuration you've trained before is instant."
)


@st.cache_data(ttl="6h", show_spinner="Loading prices, forecasts and features…")
def load_data():
    frame, _ = app_data.load_feature_data()
    return frame


# --- options (a form: nothing runs until Train) ----------------------------------------------
models = app_data.model_options()
groups = app_data.feature_group_options()
group_label = {k: label for k, label, _ in groups}
test_end = app_data.default_test_end()

with st.form("sandbox"):
    st.subheader("1. Models")
    chosen = st.multiselect(
        "Models to train",
        [m[0] for m in models],
        default=app_data.default_model_selection(app_data.DEFAULT_TEST_START, test_end),
        help="Grouped by family. The slower models take 1-2 minutes each on a long holdout "
        "the first time; after that the result is cached.",
    )

    st.subheader("2. Features")
    feature_groups = st.multiselect(
        "Extra feature groups (on top of the core set)",
        [k for k, _, _ in groups],
        format_func=group_label.get,
        help="Core: price 2/3/7/14/21/28 days earlier, the D-2 average, NESO demand and wind "
        "forecasts, residual demand, half-hour and day of week. Every group here is known at "
        "10:00 on D-1. " + " ".join(f"**{label}**: {desc}" for _, label, desc in groups),
    )
    st.caption(
        "Used by LEAR, linear quantile regression, QRA, LightGBM and the quantile forest. The "
        "naive baselines and the two time-series models have fixed inputs."
    )

    st.subheader("3. Training regime")
    c1, c2, c3 = st.columns(3)
    window = c1.selectbox(
        "Training window",
        list(app_data.WINDOW_OPTIONS),
        help="Expanding: train on all history since 2022. Rolling: only the most recent "
        "year or two, which adapts faster after a regime change (e.g. the 2022 gas crisis) "
        "but sees fewer examples. Model default: rolling for the slowest models, expanding "
        "for the rest. The time-series models always use their own short windows.",
    )
    holdout = c2.selectbox(
        "Holdout (test period)",
        list(app_data.HOLDOUT_OPTIONS),
        index=2,
        help="The delivery days forecast and scored. Shorter is quicker to train.",
    )
    refit = c3.selectbox(
        "Refit interval",
        list(app_data.REFIT_OPTIONS),
        help="Model default: weekly for quick models, every 4 weeks (13 for linear quantile "
        "regression) for slow ones. Weekly refits are slow for the slow models.",
    )

    st.subheader("4. Objective and scoring")
    c1, c2 = st.columns(2)
    objective = c1.radio(
        "Point objective",
        list(app_data.OBJECTIVE_OPTIONS),
        help="Quantiles: fit the pinball loss at 10%, 50% and 90%, giving a median and a band. "
        "Mean: fit squared error, giving a single point forecast (no band). Applies to the "
        "models that can do either; LEAR and the naive baselines are always point forecasts.",
    )
    sort_by = c2.selectbox(
        "Rank models by",
        list(METRICS),
        index=list(METRICS).index("CRPS"),
        help=" ".join(f"**{k}**: {v}" for k, v in METRICS.items()),
    )

    train = st.form_submit_button("Train", type="primary", width="stretch")

config = app_data.SandboxConfig(
    models=tuple(chosen),
    feature_groups=tuple(feature_groups),
    window=window,
    holdout=holdout,
    refit=refit,
    objective=objective,
    sort_by=sort_by,
)

if train:
    if not chosen:
        st.warning("Pick at least one model.")
    else:
        bar = st.progress(0.0, text="Starting…")
        result = app_data.run_sandbox(
            config, load_data(), lambda frac, msg: bar.progress(min(frac, 1.0), text=msg)
        )
        bar.empty()
        st.session_state["sandbox_result"] = result

result = st.session_state.get("sandbox_result")
if result is None:
    st.info("Choose your options above and press **Train** to see results.")
    st.stop()

# --- results ---------------------------------------------------------------------------------
cfg = result.config
st.divider()
st.header("Results")
st.caption(
    f"Holdout {result.test_start:%d %b %Y} to {result.test_end:%d %b %Y} · "
    f"{cfg.features.label} · {cfg.window} window · refit {cfg.refit.lower()} · {cfg.objective}"
)

st.subheader("Leaderboard")
board = result.board
st.plotly_chart(fv.fig_metric_bars(board, cfg.sort_by, colors, dark), width="stretch", theme=None)
st.dataframe(
    board.style.format(
        {
            "MAE": "£{:.2f}",
            "RMSE": "£{:.2f}",
            "Pinball": "£{:.2f}",
            "CRPS": "£{:.2f}",
            "Coverage 80%": "{:.0%}",
            "Coverage error": "{:.0%}",
            "Periods": "{:,}",
        }
    ),
    width="stretch",
)
st.markdown(
    f"Ranked by **{cfg.sort_by}**: {METRICS[cfg.sort_by]} All models are scored on exactly the "
    "same half-hours. A point forecast has no band, so its coverage is about 0% and its CRPS "
    "equals its MAE."
)

# --- model families --------------------------------------------------------------------------
st.subheader("The models")
first = next(iter(result.results.values()))
days = sorted(first["delivery_day"].unique())
c1, c2 = st.columns([2, 1])
start_day = c1.date_input(
    "Week starting",
    value=days[-7] if len(days) >= 7 else days[0],
    min_value=days[0],
    max_value=days[-1],
)
span = c2.select_slider("Days shown", options=[1, 3, 7, 14], value=7)

families = app_data.sandbox_families(result)
for tab, family in zip(st.tabs(families), families, strict=True):
    with tab:
        names = app_data.family_models(result, family)
        st.dataframe(
            board.loc[names, ["MAE", "RMSE", "CRPS", "Coverage 80%"]].style.format(
                {"MAE": "£{:.2f}", "RMSE": "£{:.2f}", "CRPS": "£{:.2f}", "Coverage 80%": "{:.0%}"}
            ),
            width="stretch",
        )
        st.plotly_chart(
            fv.fig_family_window(result.results, names, colors, dashes, start_day, span, dark),
            width="stretch",
            theme=None,
        )
        st.markdown("##### How it works")
        st.markdown(FAMILY_DOCS[family])
        st.markdown("##### Settings used")
        for name in names:
            rows = app_data.model_settings(name, cfg)
            table = "\n".join(f"| {k} | `{v}` |" for k, v in rows)
            st.markdown(f"**{name}**\n\n| Setting | Value |\n|---|---|\n{table}")

st.divider()
st.markdown(
    """
**Avoiding look-ahead.** NESO revises its forecasts many times. Every input is the latest
version published before 10:00 on D-1, never a later revision. Prices count as known only if
their half-hour ended by 09:00 on D-1, allowing an hour for publication. Tests in the repository
check this for every feature and model, including the time-series models that update on the
morning of D-1.
"""
)
