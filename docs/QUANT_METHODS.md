# Public quantitative methods and their implementation

This is a finite research toolkit, not a reconstruction of Jane Street, Citadel,
Capital One, or any other institution's proprietary models. No claim is made that
any of these institutions uses this particular implementation or combination.
A daily single-stock dataset cannot reproduce market-making systems, multi-asset
research platforms, credit underwriting, exchange connectivity or execution desks.

## Sources actually consulted online

Source retrieval was checked on October 20, 2026. The original firm pages and
several paper websites were blocked by this environment's proxy. The following
public implementation/reference sources were accessible and inspected:

| Public source | Material used |
| --- | --- |
| [arch volatility models](https://github.com/bashtage/arch/blob/main/arch/univariate/volatility.py) | EWMA/RiskMetrics recursion, GARCH family dynamics and parameter restrictions |
| [statsmodels Kalman filter](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/kalman_filter.py) | State-space filtering, recursive updates and filtered versus predicted states |
| [scikit-learn calibration](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py) | Sigmoid probability calibration; our regularized logit-on-logit calibration is a related implementation, not a copy of its estimator |
| [statsmodels HAC covariance](https://github.com/statsmodels/statsmodels/blob/main/statsmodels/stats/sandwich_covariance.py) | Newey–West covariance and Bartlett lag weights |
| [arch bootstrap](https://github.com/bashtage/arch/blob/main/arch/bootstrap/base.py) | Moving-block bootstrap and preservation of contiguous observations |
| [CVX short-course portfolio example](https://github.com/cvxgrp/cvx_short_course/blob/master/book/docs/applications/notebooks/portfolio_optimization.ipynb) | Expected portfolio return, covariance risk and Markowitz objectives |
| [SciPy statistical implementations](https://github.com/scipy/scipy/blob/main/scipy/stats/_stats_py.py) | General statistical reference context; not a source for proprietary trading equations |

[Retrieval records](research/source_checks.json) distinguish successful reads from
blocked attempts and include SHA-256 hashes for successful source downloads. The
links point to mutable source branches; the hashes identify the content observed.
Source files were read as references, not copied into this repository.

Other standard published formulas below (OU conversion, Kelly, Expected Shortfall,
Avellaneda–Stoikov, Almgren–Chriss and credit expected loss) are implemented from
established mathematical definitions. The linked original papers were **not read
online in this session** where access was blocked. They must not be described as
verified current firm practices. Original references include Kelly (1956),
Almgren & Chriss (2000), and Avellaneda & Stoikov (2008). Unread paper URLs are
listed explicitly in the retrieval records rather than presented as consulted.

## How the methods connect to the application

All new calculations live in `priceprediction/quant.py`. The existing Black–Scholes
and Monte Carlo calculations remain in `priceprediction/strategy.py`.

| Method | Role |
| --- | --- |
| EWMA, Kalman, rolling OU, return autocorrelation, downside volatility | Six causal features, used by `model_builder --quant` |
| GARCH(1,1) | Training-only fit; frozen parameters produce later volatility diagnostics |
| Moving-block Monte Carlo | Candidate order evaluation, preserving short return sequences |
| VaR and Expected Shortfall | Candidate tail-risk diagnostics and a stricter exposure cap |
| Fractional Kelly | Additional candidate notional cap; cannot enable a rejected signal |
| HAC and block-bootstrap intervals | Historical evaluation uncertainty, accounting approximately for dependence |
| Mean–variance allocation | Standalone multi-asset research helper; not used for NVDA orders |
| Inventory-aware quotes | Standalone stylized market-making helper; not sent to Alpaca |
| Optimal liquidation curve | Standalone execution-planning helper; not an order scheduler |
| PD/LGD/EAD expected loss | Standalone credit-risk arithmetic; not a stock-direction signal |

The six features are always computed causally, but existing bundles continue to
use their saved feature lists. `--quant` trains a **separate research bundle**;
it does not replace the model used by the paper workflow. Research CLI and build
API reject overwriting `artifacts/strategy.joblib` through quant mode.

## Equations

Notation: `P_t` is a completed daily close; `r_t = log(P_t/P_(t-1))`. Parameters
below are fixed defaults or estimated using training/history only. Annualization
uses 252 trading sessions. Prices, volatilities, time and impact parameters must
use mutually consistent units.

### 1. EWMA conditional volatility

\[
v_{t+1}=\lambda v_t+(1-\lambda)r_t^2,
\qquad \sigma_{ann,t+1}=\sqrt{252v_{t+1}},\quad \lambda=0.94.
\]

`q_ewma_vol` observes `r_t` at close `t` and estimates next-session volatility.
The first squared return initializes the recursion; 60 observations are required
before the feature is emitted. Returns are treated as approximately zero-mean.

### 2. GARCH(1,1) and Gaussian quasi-likelihood

\[
r_t=\mu+\epsilon_t,\quad
h_t=\omega+\alpha\epsilon_{t-1}^2+\beta h_{t-1},
\quad \omega>0,\;\alpha,\beta\ge0,\;\alpha+\beta<1.
\]

\[
\mathcal L(\theta)=\tfrac12\sum_t
\left(\log h_t+\epsilon_t^2/h_t\right).
\]

`fit_garch` estimates the mean first and minimizes the Gaussian quasi-likelihood
on demeaned training returns, rescaled to percentage units for optimization.
`garch_variance` freezes parameters and updates with newly observed returns.
This is not a Student-t model or a full likelihood fit of the mean equation.
A boundary optimizer solution is explicitly flagged; convergence alone does not
establish good statistical fit. GARCH is a diagnostic, not an added direction vote.

### 3. Scalar Kalman state-space filter

\[
x_t=x_{t-1}+w_t,\quad z_t=x_t+v_t,\quad
w_t\sim N(0,Q),\quad v_t\sim N(0,R).
\]

\[
P_t^-=P_{t-1}+Q,\quad K_t=P_t^-/(P_t^-+R),
\quad \hat x_t=\hat x_{t-1}+K_t(z_t-\hat x_{t-1}),
\quad P_t=(1-K_t)P_t^-.
\]

Here `z_t=log(P_t)`, fixed `Q=1e-5`, `R=1e-3`. The feature is
`q_kalman_gap = (log(P_t)-xhat_t)/sqrt(v_(t+1))`. This filters a latent log-price
level; it does not estimate fundamental fair value or use backward smoothing.

### 4. Rolling AR(1) / Ornstein–Uhlenbeck diagnostic

Fit a rolling 60-pair OLS regression:

\[
x_{t+1}=a+\phi x_t+\eta_t,\quad
\kappa=-\log\phi,\quad \theta=a/(1-\phi),
\quad s_\infty^2=\operatorname{mean}(\eta^2)/(1-\phi^2).
\]

For `0 < phi < 1`, `q_ou_z=(log(P_t)-theta)/s_infinity` and
`q_ou_strength=kappa`. Otherwise both OU features are zero. If interpreted as
continuous time, `dX = kappa(theta-X)dt + sigma dW` and half-life is
`log(2)/kappa` sessions. This stationarity restriction is not a unit-root test;
fitting OU to a trending price can be misleading. No pairs-trading claim is made.

### 5. Dependence and downside risk features

\[
\rho_1=\operatorname{Corr}(r_t,r_{t-1}),\qquad
\sigma_{down}=\sqrt{252\operatorname{mean}(\min(r_t,0)^2)}.
\]

Both use trailing 60-session windows. Downside volatility uses zero as its
reference return; it is not variance conditional on negative returns.

### 6. Moving-block simulation, loss VaR and Expected Shortfall

Contiguous historical blocks of length `b=5` are resampled until the five-session
path is complete. Log returns are centered and historical drift is shrunk to 25%
of its sample mean. For each path:

\[
P_{t+j}=P_t\exp\left(\sum_{k=1}^j r_k^*\right),\qquad
R= s(P_{exit}/P_t-1)-c,\quad L=-R.
\]

\[
\operatorname{VaR}_q(L)=F_L^{-1}(q),\qquad
\operatorname{ES}_q(L)=\frac{1}{1-q}\int_q^1 F_L^{-1}(u)du.
\]

The code averages the worst empirical `(1-q)N` losses, using fractional weight
at the boundary and clipping loss measures at zero. With `q=.95`, the candidate
size additionally obeys `shares * price * ES_95 <= equity * 0.005`.
This is a model-based tail budget, not a hard loss guarantee. Daily paths miss
intraday barrier crossings. A five-session block yields one historical sequence
per path; short samples and regime changes limit its usefulness.

### 7. Fractional Kelly criterion

For a stylized binary bet with win probability `p` and reward/risk `b`:

\[
f^*=p-(1-p)/b,\qquad f=\min(0.20,\max(0,0.25 f^*)).
\]

The candidate uses Monte Carlo **barrier-outcome** probability, not the model's
probability of a positive five-day close, for a conservative additional notional
cap. Monte Carlo outcomes have variable realized payoffs and gap losses, so this
binary-bet formula is only a heuristic cap, not the growth-optimal allocation
for the simulated distribution. Existing ATR, equity and notional caps remain.

### 8. Newey–West uncertainty and moving-block bootstrap

For mean-centered observations `u_t`, lag length `L`:

\[
\gamma_\ell=N^{-1}\sum_{t=\ell+1}^N u_tu_{t-\ell},
\quad \Omega=\gamma_0+2\sum_{\ell=1}^L
(1-\ell/(L+1))\gamma_\ell,\quad
SE(\bar x)=\sqrt{\max(0,\Omega)/N}.
\]

The approximate interval is `mean +/- 1.96 SE`; the candidate uses five lags.
Moving-block bootstrap intervals use the 2.5%/97.5% quantiles of 2,000 resampled
means. Reports include test accuracy and paired Brier-score improvements using
these methods. Neither approach corrects unlimited strategy searches, test-set
reuse, nonstationarity or small-sample bias.

### 9. Mean–variance portfolio research utility

\[
\max_w\;\mu^T w-\tfrac\gamma2 w^T\Sigma w,
\qquad w^*=\gamma^{-1}\Sigma^{-1}\mu.
\]

`mean_variance_weights` solves this **unconstrained** problem, permitting shorts
and leaving residual wealth in cash. It does not solve a long-only, fully invested
or transaction-cost-constrained portfolio, and does not submit stock allocations.

### 10. Inventory-aware market-making research utility

A stylized Avellaneda–Stoikov approximation:

\[
s_{reservation}=s-q\gamma\sigma^2\tau,\qquad
\delta=\gamma\sigma^2\tau+\frac{2}{\gamma}\log(1+\gamma/k),
\quad bid=s_{reservation}-\delta/2,\quad ask=s_{reservation}+\delta/2.
\]

Here `sigma` is **absolute-price** Brownian volatility, `q` is inventory and `k`
is an order-arrival liquidity parameter. These parameters are not calibrated by
daily OHLCV. `inventory_quotes` is a research function, not a production market
maker, firm model, routing algorithm or Alpaca quote placement function.

### 11. Optimal liquidation research utility

Continuous linear-impact Almgren–Chriss inventory trajectory:

\[
\kappa=\sqrt{\lambda\sigma^2/\eta},\qquad
x(t)=X\frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)},
\qquad n_i=x(t_i)-x(t_{i+1}).
\]

The zero-risk-aversion limit is uniform execution `x(t)=X(1-t/T)`.
`liquidation_schedule` samples this continuous approximation at equal time steps.
It is not the discrete-time exact solution, does not estimate market impact, and
never replaces the current five-session position manager.

### 12. Credit expected-loss arithmetic

\[
EL=PD\times LGD\times EAD.
\]

`expected_credit_loss` computes this supplied-parameter product. It does not
estimate probabilities of default, correlations or downturn LGD, and is not an
IFRS 9 lifetime ECL model or a replica of Capital One's underwriting systems.

### 13. Existing classification, calibration and option equations

The XGBoost classifier minimizes a regularized logistic objective:

\[
\hat p=\operatorname{sigmoid}(\sum_m f_m(X)),\qquad
\mathcal J=\sum_i[-y_i\log\hat p_i-(1-y_i)\log(1-\hat p_i)]
+\sum_m\Omega(f_m).
\]

Separate-period calibration fits `sigmoid(a logit(p) + b)` using regularized
logistic regression. `Brier = mean((p-y)^2)` measures probability error.
Confidence selection still uses the documented Wilson-bound heuristic and a
minimum signal count; overlapping labels mean its IID interpretation is only
approximate. HAC/bootstrap are additional evaluation diagnostics, not a replacement
qualification certificate.

Existing Black–Scholes calculations use:

\[
d_1=\frac{\log(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},
\quad d_2=d_1-\sigma\sqrt T,
\quad C=Se^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2).
\]

Put–call parity gives `P=C-S exp(-qT)+K exp(-rT)`. `Phi(d2)` is a risk-neutral
terminal probability, not a real-world trading confidence estimate. Current
scenario defaults are historical volatility and zero rates/dividends.

## Reproduce the research candidate

From the repository root with the existing environment activated:

```sh
python -m priceprediction.model_builder --quant
python -m priceprediction.backtester --bundle artifacts/research/strategy.joblib --output artifacts/research
python -m priceprediction.main --bundle artifacts/research/strategy.joblib --data data/NVDA_historical_features.csv --audit-db artifacts/research/audit.db
python -m unittest discover -s tests -v
```

No broker order or Discord notification is sent by these commands. Defaults
without `--quant` continue to use the existing model workflow. New artifacts are
ignored by Git; a compact recorded evaluation is in `research/quant_evaluation.json`.

## Observed result and limits

The candidate's historical accuracy is **58.68%**, identical to the majority-class
baseline. Brier score is **0.24920** versus **0.25878** for the retrained price-only
baseline. The paired Brier improvement's five-session block-bootstrap 95% interval
is approximately **[-0.00078, 0.02049]**, which includes zero. The previous active
model's recorded Brier score is 0.24938, so the candidate change is small and does
not establish an improvement over it. The threshold search qualifies no entries;
the candidate remains disabled and its strategy backtest executes **zero trades**.

The GARCH fit reached `alpha=0, beta=.998`, triggering a boundary-fit warning.
These volatility estimates need additional diagnostics and cleaner data before
operational use. Original data adjustment and provenance are still unverified.
This historical test period has already been inspected during earlier work;
results are exploratory, not a fresh independent test. Evaluate new forward data
before claiming an improved edge or changing the paper-trading deployment.
