# PricePrediction

A daily NVDA research strategy with a five-trading-session forecast and holding
horizon. Outputs are order proposals, with `WAIT` when evidence is insufficient.
Adding indicators does not establish a profitable or more accurate strategy.

## Project layout

```text
PricePrediction/
├── priceprediction/   # Application package, strategy, training and broker tools
├── tests/             # Offline tests and mocked broker checks
├── data/              # Cached CSV datasets and historical database
├── models/            # Supplied XGBoost model
├── reports/           # Historical HTML backtest reports
├── artifacts/         # Generated models, metrics and runtime audit DB (ignored)
├── .vscode/           # Editor workspace configuration
├── .env.example       # Credential template; keep your local .env at the root
├── requirements.txt
└── README.md
```

Run the modules below from the repository root. Shared paths are resolved relative
to the project, so data, credentials and outputs keep consistent locations.

## Run locally

Python 3.12 was used for validation. From this directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m priceprediction.model_builder
python -m priceprediction.backtester
python -m priceprediction.main --data data/NVDA_historical_features.csv
python -m unittest discover -s tests -v
```

In the prepared cloud environment, activate
`/workspace/priceprediction-venv/bin/activate` instead. Models, evaluation JSON,
and trade CSVs are saved under ignored `artifacts/`. Only load trusted local
joblib bundles: deserializing an untrusted bundle can execute code. The original
`models/nvda_xgboost_model.json` and cached data remain unchanged.

## Signals and orders

`priceprediction/strategy.py` computes the same features for training, backtesting and proposals:

- Bullish/bearish/neutral regime from SMA50/SMA200 alignment, SMA50 slope,
  directional movement and ADX. This is a causal regime classification heuristic,
  not a separately calibrated prediction of tomorrow's regime.
- Bullish/bearish engulfing candles, hammers and shooting stars.
- Support/resistance from the preceding 20 candles, excluding the current candle.
- Three-candle fair value gaps retained until full wick mitigation or 20-bar expiry.
- Technical momentum, volatility, volume and distances to market structure.

A proposal needs calibrated model confidence, agreement with the regime, a candle,
nearby level or FVG confirmation, and sufficient room to the next opposing level.
Stops are 1.5 ATR away and targets 3 ATR away. Size is capped at 0.5% of equity
in planned stop risk and 20% of equity in notional exposure. Stops do not guarantee
maximum loss when markets gap. Current support/resistance gates may reject many
setups; thresholds must not be loosened solely to obtain trades.

Monte Carlo resamples 252 trailing daily log returns over the five-session horizon,
using 4,000 reproducible paths and shrinking historical drift toward zero. It
estimates bracket returns after 0.2% round-trip cost. Entries require simulated
profit probability >=52% and positive expected net return. These probabilities
are assumptions of the simulation, not independently calibrated confidence.
Daily paths miss intraday barrier crossings and changing volatility regimes.

Black–Scholes reports European call/put values at the proposed target strike and
risk-neutral terminal probability above that strike. It uses trailing realized
volatility, five-session maturity, and zero rates/dividends. These are scenario
estimates, not option quotes or real-world directional probabilities. They are
reported alongside proposals and do not vote on stock direction. Real option
pricing would require expiry, strike, rates, dividends and option-market inputs.

## Training and evidence

Features use completed candles only. Labels explicitly compare the close five
sessions ahead; unlabeled final rows are excluded. Chronological train/calibration/
validation/test partitions are 50/20/15/15%, with five-session purges at all three
boundaries. Logistic probability calibration uses only the calibration period.
The validation period selects a symmetric confidence cutoff from 0.50–0.80.
Selection requires >=25 regime-qualified signals, >=55% directional precision,
and a 95% Wilson lower bound above 50%. These are exploratory criteria: overlapping
labels and threshold searches mean the bound is not a formal trading guarantee.
No qualifying threshold means trading is disabled, rather than selecting a
threshold after looking at the final test results. Tests remain separate from
calibration and validation, and no test-period refit occurs.

After merging the updated dataset from `main`, the evaluation produced:

| Holdout measure | Result |
| --- | ---: |
| Test sessions with labels | 317 |
| Calibrated directional accuracy | 58.68% |
| Price-only baseline accuracy on the same split/horizon | 53.00% |
| Always-bullish accuracy | 58.68% |
| Calibrated Brier score | 0.24938 |
| Baseline Brier score | 0.25878 |
| Qualified validation threshold | None |
| Strategy trades | 0 |

The model has not demonstrated an edge over the majority-class prediction.
Zero trades validate abstention, not strategy profitability or trade accuracy.
`artifacts/strategy.report.json` records splits, threshold candidates and results;
`artifacts/backtest.json` and `trades.csv` record execution evaluation. The new
price-only baseline is retrained on the same split; it is not the old saved model.
Cached sentiment is excluded because its historical availability is unverified.

The backtest executes signals at the next open, with 0.1% commission per side,
no leverage and a five-session time exit. It does not inspect the next opening
price to decide whether to enter. Opening gaps can invalidate planned risk/target
geometry. The engine resolves ambiguous daily intrabar fills conservatively where
possible; borrow costs, funding, spread and market impact are not modeled.
Historical adjustment/timestamp provenance still needs verification before using
these results as investment evidence.

## Optional paper workflow

Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` securely (or the existing
`APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` names from `.env.example`), then run `python -m priceprediction.main`
for a fresh-data proposal only. The old hardcoded keys have been removed from the
Python files; rotate the previously exposed credentials, which remain in Git
history. Do not commit credentials or `.env` files. No Hugging Face downloads or
FinBERT initialization are needed for this price-only strategy.

`python -m priceprediction.main --submit-paper` is an explicit paper-only execution command.
No real-money endpoint is supported. Run once each open-market session if enabled:
this command also checks managed positions for the five-session time exit.
It discards today's potentially incomplete candle and refuses stale signals,
unmanaged positions, duplicate/pending entries and unshortable shorts. Paper entry
uses a DAY limit at the signal close plus bracket exits, so fills can differ from
the market-entry research backtest. The deterministic client order ID prevents
reusing a daily signal. Time exits require this command to keep running daily;
they are not enforced by the broker bracket alone. Partial fills or uncertain
position ownership require manual review rather than automatic liquidation.

Paper broker integration is covered by mocked tests; no API authentication,
market-data feed entitlement, paper fills or order placement was tested here.
The current trained bundle will abstain until a newly trained model qualifies on
its own validation data. Do not edit the bundle to bypass that result.

## Integration with the updated main branch

The updated cached dataset, supplied model, credential configuration, SQLite audit
module, notification module and retry/position-size utilities are retained.
`priceprediction/config.py` loads either credential naming convention from the environment or a
local `.env` file. Market-data reads use the existing retry helper; broker writes
are not blindly retried. The strategy keeps its stricter position-size caps.

Proposal audits and submitted-entry audits use `artifacts/trading_bot.db` by
default (`--audit-db PATH` to change it), leaving the tracked historical
`data/trading_bot.db` intact. Offline CSV proposals are logged locally too. Discord
alerts require explicit `--notify` plus a configured `DISCORD_WEBHOOK_URL`;
normal evaluation and tests never send notifications. Paper broker paths have been tested with mocks only; Discord delivery has not
been exercised.

## Public quant research toolkit

The mathematical reference, source-access record, equations and implementation
roles are documented in [docs/QUANT_METHODS.md](docs/QUANT_METHODS.md). These are
public methods, not replicas of proprietary Jane Street, Citadel or Capital One
models. The toolkit adds EWMA volatility, Kalman filtering, OU diagnostics,
GARCH fitting, moving-block simulation, tail-risk measures, fractional Kelly,
HAC/bootstrap uncertainty and standalone portfolio/execution/credit calculations.

Train the optional candidate with `python -m priceprediction.model_builder --quant`.
It is saved separately under `artifacts/research/`; the active paper bundle is
preserved. The current research candidate has no qualified confidence threshold
and does not demonstrate an improved directional edge. See the reference for
reproduction commands, results and limitations.
