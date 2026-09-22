# adaptive-etf-rotation
 Transparent multi-asset ETF rotation using multi-horizon momentum, trend filters, rank, hysteresis, inverse-volatility sizing, and volatility targeting.

Install with `poetry install --no-root`. Set `ALPACA_KEY_ID`,
`ALPACA_SECRET_KEY`, and `ALPACA_PAPER` (defaults to `true`) in the environment
or `.env`.

Strategy settings are read in `algo.py` and passed as `config` to
`run_single_iteration`. Environment variables and their defaults:

```dotenv
ETF_UNIVERSE=QQQ,EFA,TLT,GLD,VNQ
CASH_ETF=BIL
HISTORY_START=2006-01-01
TOP_N=3
EXIT_RANK=5
MOMENTUM_LOOKBACKS=63,126,252
MOMENTUM_WEIGHTS=0.50,0.30,0.20
TREND_MA=200
VOL_LOOKBACK=20
TARGET_VOL=0.20
MAX_GROSS_EXPOSURE=1.50
HIGH_VOL_ADJUSTMENT_ENABLED=true
HIGH_VOL_THRESHOLD=1.20
HIGH_VOL_WEIGHT_MULTIPLIER=0.70
HIGH_VOL_REFERENCE_LOOKBACK=252
USE_EMA_SMOOTHING=true
EMA_ALPHA=0.30
USE_HYSTERESIS=true
EQUITY_FRACTION=1.0
APP_STATE=OBSERVE
FORCED_REBALANCE=false
```

Lists use comma-separated values. Booleans accept `true/false`, `yes/no`, or
`1/0`. Existing environment variables take precedence over `.env`. CLI options
override environment defaults; `--no-force-rebalance` disables forced rebalancing.
`FORCE_REBALANCE` remains a fallback alias for `FORCED_REBALANCE`.
`STATE_PATH` optionally overrides the journal path.

Observe: `APP_STATE=OBSERVE poetry run python algo.py --equity-fraction 0.5`

Submit to the configured account: `APP_STATE=LIVE poetry run python algo.py --equity-fraction 0.5`

`APP_STATE` must be `LIVE`, `PAPER`, or `OBSERVE` (`RUN_MODE` is a fallback).
Following the reporting workflow, only LIVE submits orders. `ALPACA_PAPER`
independently selects the broker endpoint: LIVE with ALPACA_PAPER=true submits
to Alpaca's paper account. PAPER simulates allocations and persists monthly
completion in a separate default simulation journal without submitting orders.
OBSERVE changes no journal files and signals trading only on the first NYSE
session of the month, unless forced. Off-schedule observations still calculate
current allocations for reports while retaining `trade_today=false`.

Set `EMAIL_POSITIONS=true` to send HTML reports using `SES.py`. Configure
`FROM_ADDRESS`, comma-separated `TO_ADDRESSES`, `AWS_SES_REGION_NAME`,
`AWS_SES_ACCESS_KEY_ID`, and `AWS_SES_SECRET_ACCESS_KEY`.

Set `SYNC_STRATEGY_JSON_TO_SPACES=true` to export `etf-adaptive-rotation.json`
and upload it using the existing Spaces helper. Configure `SPACES_REGION`,
`SPACES_BUCKET`, `SPACES_KEY`, and `SPACES_SECRET`; `SPACES_OBJECT_KEY_PATH`
optionally prefixes the filename. Both integrations default to disabled.
`.example.env` includes the settings and credential placeholders.

`algo.py` constructs the adapter and reports portfolio value.
`helpers.run_single_iteration(api, equity_fraction=1.0, is_live_trade=False,
force_rebalance=False)` implements the notebook strategy. The fraction applies
to total account equity. Strategy-symbol holdings are managed in full; other
symbols are left alone. Do not share strategy symbols with another strategy.

By default, the first invocation during market hours each month uses the previous
month's final available session. `--force-rebalance` bypasses the monthly gate
and uses the latest completed session. Preview runs do not mark a month traded.
Alpaca adjusted daily bars drive signals; unadjusted recent closes estimate order
quantities. Results can differ from the notebook's Yahoo data and simulated fills.
The notebook's 1.5 exposure cap is preserved; negative cash is borrowing, not a
short BIL position. Available buying power must cover orders.

Keep the account-specific `.rotation-state-*.json` file on persistent storage
(or supply `--state-path`). It records monthly completion and selection history.
Orders execute sells first and wait for fills. A failure or timeout leaves a
pending journal and blocks further submissions, including forced runs. Reconcile
the journal against broker orders and positions before clearing its `pending`
field; submission may have succeeded even if the response was interrupted.

Run checks with `poetry run python -m unittest discover -s tests -v`.
