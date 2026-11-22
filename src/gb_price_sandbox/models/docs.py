r"""Plain-English and mathematical descriptions of each model family, for the Forecast sandbox.

Markdown with LaTeX (``$...$`` inline, ``$$...$$`` display), rendered by Streamlit. Every symbol
is defined where it is introduced; :data:`NOTATION` holds the shared ones.
"""

from __future__ import annotations

NOTATION = r"""
**Notation.** $p_{d,h}$ is the price (£/MWh) on delivery day $d$ in half-hour $h \in \{1,\dots,48\}$,
and $\hat p_{d,h}$ its forecast. Where a model works on a transformed scale,
$\tilde p = \operatorname{asinh}\!\big((p - m)/s\big)$, with $m$ the median and $s$ the median
absolute deviation of training prices: linear near the middle, logarithmic in both tails.
$\tau \in \{0.1, 0.5, 0.9\}$ is a quantile level and $\hat q_\tau$ the forecast of that quantile.
"""

PINBALL = r"""
**Pinball loss.** For level $\tau$ and error $u = p - \hat q_\tau$,

$$
\rho_\tau(u) = \max\big(\tau\,u,\; (\tau - 1)\,u\big).
$$

Minimising its average gives the $\tau$-quantile: at $\tau = 0.9$, forecasting too low costs nine
times as much as forecasting too high, so the forecast settles where 90% of outcomes fall below it.
"""

FAMILY_DOCS: dict[str, str] = {
    "Naive baselines": r"""
Three rules with nothing to fit:

$$
\underbrace{\hat p_{d,h} = p_{d-7,h}}_{\text{seasonal naive}} \qquad
\underbrace{\hat p_{d,h} = p_{d-2,h}}_{\text{D-2 persistence}} \qquad
\underbrace{\hat p_{d,h} = \tfrac14 \textstyle\sum_{k=1}^{4} p_{d-7k,\,h}}_{\text{weekday profile}}
$$

Day $d-2$ is the latest full day known at 10:00 on $d-1$, so persistence tracks the price level
but forecasts Monday from Saturday. The seasonal naive gets the weekday right but uses week-old
weather; the weekday profile averages four weeks of it. All three are point forecasts, so the
feature-set and objective options don't apply.
""",
    "LEAR": r"""
**LASSO-estimated autoregression** (Lago et al., 2021). One linear model per half-hour $h$:

$$
\tilde p_{d,h} = \beta_{0,h} + \beta_h^\top x_{d,h} + \varepsilon_{d,h},
$$

where $x_{d,h}$ stacks the asinh prices $\tilde p_{d-2,h}, \tilde p_{d-3,h}, \tilde p_{d-7,h}$ and the
D-2 daily mean, the standardised demand and wind forecasts, six weekday dummies, and any extra
feature groups; $\varepsilon_{d,h}$ is the error. The coefficients minimise

$$
\sum_{d} \big(\tilde p_{d,h} - \beta_{0,h} - \beta_h^\top x_{d,h}\big)^2 + \lambda_h \lVert\beta_h\rVert_1 ,
$$

whose L1 penalty (strength $\lambda_h$, chosen by AIC at each refit) sets unhelpful coefficients
exactly to zero. The forecast is transformed back from the asinh scale. It is a mean model, so it
gives a point forecast whatever the objective.
""",
    "MSTL": r"""
**Seasonal decomposition plus an autoregression.** Over the last 28 days the asinh price series
$\tilde p_t$ (indexed by half-hour $t$) is split (MSTL; Bandara et al., 2021) as

$$
\tilde p_t = T_t + S^{\text{day}}_t + S^{\text{week}}_t + R_t ,
$$

a slow trend $T_t$, a daily pattern $S^{\text{day}}_t$ (period 48), a weekly pattern
$S^{\text{week}}_t$ (period 336) and a remainder $R_t$. The patterns are projected forward by
repeating their latest cycle; $y_t = T_t + R_t$ follows an AR(2),

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal N(0,\sigma^2),
$$

with constant $c$, coefficients $\phi_1, \phi_2$ and noise variance $\sigma^2$. Parameters are
re-estimated every 4 weeks; each morning the model is updated with prices published by 09:00 on
D-1 and run forward. The band is the 80% Gaussian interval on the asinh scale, which becomes
skewed in £/MWh. Inputs and window are fixed, so those options don't apply.
""",
    "SARIMAX": r"""
**Regression with ARMA errors.** On the last 14 days of the asinh price $\tilde p_t$:

$$
\tilde p_t = \beta_0 + \sum_{k=1}^{3} \big[a_k \sin 2\pi k u_t + b_k \cos 2\pi k u_t\big]
  + \sum_{k=1}^{2} \big[c_k \sin 2\pi k w_t + d_k \cos 2\pi k w_t\big] + \gamma^\top z_t + \eta_t ,
$$

where $u_t, w_t \in [0, 1)$ are the time of day and of week (local clock), $z_t$ the standardised
demand and wind forecasts, and $\beta_0, a_k, b_k, c_k, d_k, \gamma$ coefficients. The error
$\eta_t$ follows an ARMA(1, 1),

$$
\eta_t = \phi\,\eta_{t-1} + \varepsilon_t + \theta\,\varepsilon_{t-1}, \qquad \varepsilon_t \sim \mathcal N(0,\sigma^2),
$$

so today's surprise carries into tomorrow. The Fourier terms stand in for seasonal ARIMA terms
with 48- and 336-period seasons, which would be far too slow to fit. Parameters are estimated by
maximum likelihood every 4 weeks; each morning the Kalman filter is re-run on the latest prices.
The band is the 80% Gaussian interval on the asinh scale.
""",
    "Quantile regression": r"""
**Linear quantile regression** (Koenker and Bassett, 1978). LEAR's inputs $x_{d,h}$, one model per
half-hour $h$ and level $\tau$, with no penalty:

$$
\hat\beta_{\tau,h} = \arg\min_{\beta} \sum_d \rho_\tau\big(\tilde p_{d,h} - \beta^\top x_{d,h}\big),
\qquad \hat q_{\tau} = \text{back-transform of } \hat\beta_{\tau,h}^\top x_{d,h} .
$$

Quantiles survive monotone transforms, so back-transforming from the asinh scale still gives
the $\tau$-quantile of the price. That is $48 \times 3$ small linear programs per refit, so by
default it refits quarterly on the last year. Crossed quantiles are sorted, which never worsens the
pinball loss. With the **mean** objective, $\rho_\tau$ becomes squared error (ordinary least squares).
"""
    + PINBALL,
    "QRA": r"""
**Quantile regression averaging** (Nowotarski and Weron, 2015). Four point forecasts
$f^{(1)}, \dots, f^{(4)}$ (LEAR, seasonal naive, D-2 persistence, weekday profile) are combined by
one quantile regression per level, pooled over half-hours:

$$
\hat q_\tau = w_{0,\tau} + \sum_{j=1}^{4} w_{j,\tau}\, f^{(j)},
\qquad \hat w_\tau = \arg\min_{w} \sum_{t \in \mathcal C} \rho_\tau\big(p_t - \hat q_\tau(t)\big),
$$

where the weights $w_{j,\tau}$ are fitted on a calibration set $\mathcal C$, the last 182 days of
training. They must be learnt from genuine out-of-sample forecasts, so LEAR is replayed over
$\mathcal C$, refitted every 14 days on data up to two days before. The weights say how far to
trust each input; the gap between $\hat q_{0.1}$ and $\hat q_{0.9}$ says how wrong the inputs have
typically been. With the **mean** objective the weights are fitted by least squares instead.
"""
    + PINBALL,
    "LightGBM": r"""
**Gradient-boosted trees.** The forecast for input vector $x$ is a sum of $M$ small regression trees,

$$
\hat q_\tau(x) = \sum_{m=1}^{M} \nu\, f_m(x),
$$

where each tree $f_m$ maps $x$ to a constant per leaf and $\nu$ is the learning rate. Trees are
added one at a time, each fitted to the negative gradient of the loss at the current sum. For
$\rho_\tau$ that gradient is simply $\tau$ or $\tau - 1$ depending on the sign of the error, so each
tree nudges the forecast towards the quantile. One ensemble per $\tau$, or one with squared error
for the **mean** objective.

A single model covers all half-hours, with $h$ and the weekday among the inputs, so trees find
interactions (wind matters more overnight) unprompted. Inputs need no scaling: trees use only their
order. Settings are fixed and small ($M = 100$, $\nu = 0.1$, 15 leaves), for speed.
"""
    + PINBALL,
    "Quantile forest": r"""
**Quantile regression forest** (Meinshausen, 2006). $B$ trees are grown, each on a random sample of
training rows $i$ with inputs $x_i$ and price $p_i$. For a new input $x$, let $L_b(x)$ be the set of
training rows sharing its leaf in tree $b$. Each training row gets the weight

$$
w_i(x) = \frac{1}{B}\sum_{b=1}^{B} \frac{\mathbb 1\big[i \in L_b(x)\big]}{\lvert L_b(x)\rvert},
$$

and the forecast distribution and its quantiles are

$$
\hat F(y \mid x) = \sum_i w_i(x)\,\mathbb 1[p_i \le y], \qquad
\hat q_\tau(x) = \min\{\,y : \hat F(y \mid x) \ge \tau\,\}.
$$

One forest gives every quantile, and the band reflects how spread out prices were in similar past
situations, with no assumed shape. With the **mean** objective it reports $\sum_i w_i(x)\,p_i$,
an ordinary random forest's prediction.
""",
}
