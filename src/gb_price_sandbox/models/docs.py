r"""Plain-English and mathematical descriptions of each model family, for the Forecast sandbox.

Markdown with LaTeX (``$...$`` inline, ``$$...$$`` display), rendered by Streamlit. Notation used
throughout: $p_{d,h}$ is the price on delivery day $d$ in half-hour $h$; $\tilde p$ is a price
on LEAR's asinh scale; $x_{d,h}$ are the inputs; $\tau \in \{0.1, 0.5, 0.9\}$ is a quantile
level.
"""

from __future__ import annotations

PINBALL = r"""
The **pinball loss** for quantile level $\tau$ is

$$
\rho_\tau(u) = \begin{cases} \tau\,u & u \ge 0 \\ (\tau - 1)\,u & u < 0 \end{cases}
\qquad u = \text{actual} - \text{forecast}.
$$

Minimising it gives the $\tau$-quantile: at $\tau = 0.9$ under-forecasting costs nine times as
much as over-forecasting, so the fit settles where 90% of outcomes fall below it.
"""

FAMILY_DOCS: dict[str, str] = {
    "Naive baselines": r"""
Three rules with nothing to fit. They set the bar every learning model has to clear.

$$
\begin{aligned}
\text{Seasonal naive:}\quad & \hat p_{d,h} = p_{d-7,h} \\
\text{D-2 persistence:}\quad & \hat p_{d,h} = p_{d-2,h} \\
\text{Weekday profile:}\quad & \hat p_{d,h} = \tfrac14\left(p_{d-7,h} + p_{d-14,h}
    + p_{d-21,h} + p_{d-28,h}\right)
\end{aligned}
$$

D-2 is the most recent full day known at 10:00 on D-1, so persistence tracks the price level
best but forecasts Monday from Saturday. The seasonal naive gets the weekday right but uses
week-old weather. The weekday profile averages four weeks, trading reaction speed for less
noise. All three are point forecasts, so they have no band; the feature set and objective
options don't apply.
""",
    "LEAR": r"""
**LASSO-estimated AutoRegressive model** (Lago et al., 2021), a standard benchmark in electricity
price forecasting. One linear model per half-hour $h$, on the asinh scale:

$$
\tilde p_{d,h} = \beta_{0,h} + \sum_{k \in \{2,3,7\}} \beta_{k,h}\,\tilde p_{d-k,h}
  + \beta_{m,h}\,\overline{\tilde p}_{d-2} + \gamma_h^\top z_{d,h}
  + \sum_{j=1}^{6} \delta_{j,h}\,\mathbb{1}[\text{weekday}_d = j] + \varepsilon_{d,h}
$$

where $z_{d,h}$ are the standardised demand and wind forecasts plus any extra feature groups.
The transform is $\tilde p = \operatorname{asinh}\!\big((p - \text{median}) / \text{MAD}\big)$,
which behaves like a straight line near the middle and a logarithm in the tails, so price
spikes and negative prices don't dominate. Coefficients minimise

$$
\sum_d \big(\tilde p_{d,h} - \hat{\tilde p}_{d,h}\big)^2 + \lambda_h \lVert \beta_h \rVert_1 ,
$$

and the L1 penalty switches off unhelpful inputs. $\lambda_h$ is chosen by AIC at every refit.
LEAR is a mean model, so it gives a point forecast whatever the objective.
""",
    "MSTL": r"""
**Seasonal decomposition plus an autoregression.** Over the last 28 days the asinh price is
split (MSTL: Loess-based, Bandara et al., 2021) into

$$
\tilde p_t = T_t + S^{(48)}_t + S^{(336)}_t + R_t ,
$$

a slow trend, a daily pattern (48 half-hours), a weekly pattern (336) and a remainder. The
seasonal patterns are projected forward by repeating their latest cycle. Trend plus remainder,
$y_t = T_t + R_t$, follows an AR(2):

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \varepsilon_t, \qquad
\varepsilon_t \sim \mathcal N(0, \sigma^2).
$$

The parameters are refitted every 4 weeks. Each day the model is updated with the prices
published by 09:00 on D-1 (without re-estimating) and forecast up to 39 hours ahead. The band
is the AR model's Gaussian 80% interval on the asinh scale, which becomes skewed in £/MWh.
It uses no demand or wind information, and its own fixed window, so the feature set and
window options don't apply.
""",
    "SARIMAX": r"""
**Regression with ARIMA errors.** On the last 14 days of the asinh price:

$$
\tilde p_t = \beta_0 + \sum_{k=1}^{3}\big[a_k \sin(2\pi k\,u_t) + b_k \cos(2\pi k\,u_t)\big]
  + \sum_{k=1}^{2}\big[c_k \sin(2\pi k\,w_t) + d_k \cos(2\pi k\,w_t)\big]
  + \gamma^\top z_t + \eta_t ,
$$

where $u_t$ is the time of day and $w_t$ the time of week (local clock, both scaled to
$[0,1)$), $z_t$ the standardised demand and wind forecasts, and the error follows an ARMA(1, 1):

$$
\eta_t = \phi\,\eta_{t-1} + \varepsilon_t + \theta\,\varepsilon_{t-1}.
$$

A true seasonal ARIMA with a 336-period season would be far too slow to fit, so the seasons go
into the regression as **Fourier terms**. Parameters are estimated by maximum likelihood (Kalman
filter) every 4 weeks; each day the filter is re-run on the latest published prices and the
forecast carried forward. The band is the Gaussian 80% interval on the asinh scale.
""",
    "Quantile regression": r"""
**Linear quantile regression** (Koenker and Bassett, 1978): LEAR's inputs, but fitted for each
quantile separately and without a penalty. For each half-hour $h$ and level $\tau$:

$$
\hat\beta_{\tau,h} = \arg\min_\beta \sum_d \rho_\tau\!\big(\tilde p_{d,h} - x_{d,h}^\top\beta\big).
$$

Because quantiles survive monotone transforms, transforming $x^\top\hat\beta_\tau$ back from the
asinh scale gives the $\tau$-quantile of the price. That is 144 small linear programs per refit
(48 half-hours × 3 quantiles), so by default it refits quarterly on the last year. If the
fitted quantiles cross, they are sorted (which never makes the pinball loss worse). With the
**mean** objective the pinball loss is replaced by squared error, i.e. ordinary least squares.
"""
    + PINBALL,
    "QRA": r"""
**Quantile Regression Averaging** (Nowotarski and Weron, 2015) combines several point forecasts
$f^{(1)}, \dots, f^{(4)}$ (LEAR, seasonal naive, D-2 persistence, weekday profile) with one
quantile regression per level, pooled across half-hours:

$$
\hat q_\tau = w_{0,\tau} + \sum_{j=1}^{4} w_{j,\tau}\, f^{(j)}, \qquad
\hat w_\tau = \arg\min_w \sum_{t \in \text{cal}} \rho_\tau\!\big(p_t - \hat q_\tau(t)\big).
$$

The weights must be learnt from genuine out-of-sample forecasts. So each refit replays a mini
walk-forward over the last 182 days, refitting LEAR every 14 days on data up to D-2. The
regression is fitted on those forecasts, then LEAR is refitted on everything for the days
ahead. The weights show how far to trust each input, and the spread between $\hat q_{0.1}$ and
$\hat q_{0.9}$ shows how wrong the point forecasts have typically been. With the **mean**
objective the weights are fitted by least squares (classic forecast averaging).
"""
    + PINBALL,
    "LightGBM": r"""
**Gradient-boosted trees.** The forecast is a sum of $M$ small regression trees,

$$
\hat q_\tau(x) = \sum_{m=1}^{M} \nu\, f_m(x),
$$

built one at a time: tree $f_m$ is fitted to the negative gradient of the loss at the current
ensemble, and $\nu$ is the learning rate. With the pinball loss $\rho_\tau$ the gradient is
just $\tau$ or $\tau - 1$ depending on the sign of the error, so each tree nudges the forecast
up or down towards the quantile. One ensemble per quantile, or a single one with squared error
for the **mean** objective.

One model covers all 48 half-hours, with the half-hour and weekday as inputs, so trees find
interactions (e.g. wind matters more overnight) without being told. Inputs are used raw: trees
only care about ordering, so no asinh or scaling is needed. Settings are fixed and small, for
speed.
"""
    + PINBALL,
    "Quantile forest": r"""
**Quantile regression forest** (Meinshausen, 2006). A random forest grows $B$ trees, each on a
random sample of rows. For a new input $x$, let $L_b(x)$ be the leaf it lands in for tree $b$.
Each training row $i$ gets a weight

$$
w_i(x) = \frac{1}{B}\sum_{b=1}^{B} \frac{\mathbb 1[x_i \in L_b(x)]}{\lvert L_b(x) \rvert},
$$

and the forecast distribution is the weighted empirical distribution of the training prices:

$$
\hat F(y \mid x) = \sum_i w_i(x)\,\mathbb 1[p_i \le y], \qquad
\hat q_\tau(x) = \inf\{y : \hat F(y \mid x) \ge \tau\}.
$$

So one forest gives every quantile, and the band reflects how spread out prices were in
genuinely similar past situations, with no assumed shape. With the **mean** objective it
reports $\sum_i w_i(x)\,p_i$, which is an ordinary random forest's prediction.
""",
}
