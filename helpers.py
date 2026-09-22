"""Signal calculations ported from diversified_etf_rotation_backtest.ipynb."""

from __future__ import annotations

import fcntl
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

import numpy as np
import pandas as pd


def str2bool(value):
    valid = {
        "true": True,
        "t": True,
        "1": True,
        "on": True,
        "yes": True,
        "false": False,
        "f": False,
        "0": False,
        "off": False,
        "no": False,
    }

    if isinstance(value, bool):
        return value

    lower_value = str(value).strip().lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


def getenv_float(name: str, default: float) -> float:
    """
    Read an environment variable as a float.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def getenv_int(name: str, default: int) -> int:
    """
    Read an environment variable as an integer.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def is_paper_account():
    """Read ALPACA_PAPER; default to the paper endpoint."""
    return str2bool(os.getenv("ALPACA_PAPER", True))


def get_app_state():
    state = (os.getenv("APP_STATE") or os.getenv("RUN_MODE") or "").strip().upper()
    if state not in {"LIVE", "PAPER", "OBSERVE"}:
        raise ValueError("APP_STATE must be LIVE, PAPER, or OBSERVE.")
    return state


def monthly_rebalance_day(date):
    import pandas_market_calendars as mcal

    date = pd.Timestamp(date).tz_localize(None).normalize()
    start = date.replace(day=1)
    sessions = mcal.get_calendar("NYSE").valid_days(
        start, start + pd.offsets.MonthEnd(0)
    )
    return sessions[0].tz_localize(None)


def result_trade_today(result):
    return bool(
        result.get("meta", {}).get("trade_today", result.get("status") == "rebalanced")
    )


def format_observe_allocations(weights):
    if not isinstance(weights, dict):
        return []
    entries = [(s, float(w)) for s, w in weights.items() if abs(float(w)) > 1e-12]
    return [
        f"{s}: {w * 100:.2f}%"
        for s, w in sorted(entries, key=lambda item: item[1], reverse=True)
    ]


def build_strategy_snapshot_for_reporting(
    portfolio, *, api, config, equity_fraction, state_path=None
):
    if portfolio.get("weights"):
        return portfolio
    snapshot = run_single_iteration(
        api,
        config=config,
        equity_fraction=equity_fraction,
        state_path=state_path,
        force_rebalance=True,
        is_live_trade=False,
        persist_state=False,
    )
    meta = dict(snapshot.get("meta", {}))
    for key in ("reason", "scheduled_rebalance_day", "date", "trade_today"):
        if key in portfolio.get("meta", {}):
            meta[key] = portfolio["meta"][key]
    return {**snapshot, "meta": meta, "orders": []}


@dataclass(frozen=True)
class StrategyConfig:
    universe: tuple = ("QQQ", "EFA", "TLT", "GLD", "VNQ")
    cash: str = "BIL"
    history_start: str = "2006-01-01"
    top_n: int = 3
    exit_rank: int = 5
    momentum_lookbacks: tuple = (63, 126, 252)
    momentum_weights: tuple = (0.50, 0.30, 0.20)
    trend_ma: int = 200
    vol_lookback: int = 20
    target_vol: float = 0.20
    max_gross_exposure: float = 1.50
    high_vol_adjustment_enabled: bool = True
    high_vol_threshold: float = 1.20
    high_vol_weight_multiplier: float = 0.70
    high_vol_reference_lookback: int = 252
    use_ema_smoothing: bool = True
    ema_alpha: float = 0.30
    use_hysteresis: bool = True


def validate_config(cfg):
    if (
        not cfg.universe
        or len(set(cfg.universe)) != len(cfg.universe)
        or cfg.cash in cfg.universe
    ):
        raise ValueError("Require distinct risky ETFs and a separate cash ETF.")
    if not cfg.momentum_lookbacks or len(cfg.momentum_lookbacks) != len(
        cfg.momentum_weights
    ):
        raise ValueError(
            "Momentum lookbacks and weights must have equal nonzero length."
        )
    if not np.isclose(sum(cfg.momentum_weights), 1) or any(
        w < 0 for w in cfg.momentum_weights
    ):
        raise ValueError("Momentum weights must be nonnegative and sum to one.")
    if any(
        not isinstance(n, int) or n < 1
        for n in (*cfg.momentum_lookbacks, cfg.trend_ma, cfg.top_n, cfg.exit_rank)
    ):
        raise ValueError("Lookbacks and selection counts must be positive integers.")
    if cfg.vol_lookback < 2 or cfg.high_vol_reference_lookback < cfg.vol_lookback:
        raise ValueError("Require reference lookback >= volatility lookback >= 2.")
    if cfg.exit_rank < cfg.top_n or not 0 < cfg.ema_alpha <= 1:
        raise ValueError("Require exit_rank >= top_n and 0 < ema_alpha <= 1.")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (cfg.target_vol, cfg.max_gross_exposure, cfg.high_vol_threshold)
    ):
        raise ValueError(
            "Volatility target, exposure cap, and threshold must be positive and finite."
        )
    if not 0 < cfg.high_vol_weight_multiplier <= 1:
        raise ValueError("High-volatility multiplier must be in (0, 1].")


def normalize_prices(prices, symbols, today):
    """Keep only completed sessions, never fill missing observations."""
    prices = prices.reindex(columns=list(symbols)).copy()
    index = pd.DatetimeIndex(pd.to_datetime(prices.index))
    if index.tz is not None:
        index = index.tz_convert("America/New_York").tz_localize(None)
    prices.index = index.normalize()
    prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.where(np.isfinite(prices) & prices.gt(0))
    return prices.loc[prices.index < today].dropna(how="all")


def load_prices(api, symbols, start, end, adjustment="all", feed="iex"):
    from alpaca.data.timeframe import TimeFrame

    bars = api.get_bars(
        list(symbols),
        TimeFrame.Day,
        start=start,
        end=end,
        adjustment=adjustment,
        feed=feed,
    )
    frame = bars.df
    if frame.empty:
        raise ValueError("Alpaca returned no daily bars.")
    if isinstance(frame.index, pd.MultiIndex):
        return frame["close"].unstack("symbol")
    if len(symbols) == 1:
        return frame[["close"]].rename(columns={"close": symbols[0]})
    raise ValueError("Expected symbol/timestamp indexed Alpaca bars.")


def prepare_research_data(prices, cfg):
    risky = prices.loc[:, list(cfg.universe)]
    raw = sum(
        weight * risky.pct_change(lookback, fill_method=None)
        for lookback, weight in zip(cfg.momentum_lookbacks, cfg.momentum_weights)
    )
    smoothed = raw.ewm(alpha=cfg.ema_alpha, adjust=False, min_periods=1).mean()
    ranking = smoothed if cfg.use_ema_smoothing else raw.copy()
    moving_average = risky.rolling(cfg.trend_ma, min_periods=cfg.trend_ma).mean()
    trend = risky.gt(moving_average) & risky.notna()
    returns = risky.pct_change(fill_method=None)
    volatility = returns.rolling(
        cfg.vol_lookback, min_periods=cfg.vol_lookback
    ).std() * np.sqrt(252)
    reference = (
        volatility.rolling(
            cfg.high_vol_reference_lookback, min_periods=cfg.vol_lookback
        )
        .median()
        .shift(1)
    )
    ratio = volatility.div(reference.replace(0, np.nan))
    trigger = ratio.ge(cfg.high_vol_threshold).fillna(False)
    if not cfg.high_vol_adjustment_enabled:
        trigger.loc[:, :] = False
    return {
        "momentum_score": raw,
        "smoothed_score": smoothed,
        "ranking_score": ranking,
        "moving_average": moving_average,
        "trend_filter": trend,
        "eligibility": trend & ranking.notna() & volatility.notna() & risky.notna(),
        "risky_returns": returns,
        "realized_volatility": volatility,
        "high_vol_reference": reference,
        "high_vol_ratio": ratio,
        "high_vol_trigger": trigger,
    }


def select_etfs(research, date, previous, cfg):
    eligible = [s for s in cfg.universe if research["eligibility"].at[date, s]]
    scores = (
        research["ranking_score"]
        .loc[date, eligible]
        .dropna()
        .sort_values(ascending=False, kind="mergesort")
    )
    ranks = {s: i + 1 for i, s in enumerate(scores.index)}
    if cfg.use_hysteresis:
        retained = sorted(
            (
                s
                for s in dict.fromkeys(previous)
                if ranks.get(s, math.inf) <= cfg.exit_rank
            ),
            key=ranks.get,
        )
        entrants = [
            s for s in scores.index if ranks[s] <= cfg.top_n and s not in retained
        ]
        selected = (retained + entrants)[: cfg.top_n]
    else:
        selected = list(scores.index[: cfg.top_n])
    return selected, ranks, scores.to_dict()


def calculate_weights(selected, date, research, cfg):
    """Same inverse-volatility, covariance scaling and haircut as the notebook."""
    window = research["risky_returns"].loc[:date, list(selected)].tail(cfg.vol_lookback)
    valid = [
        s
        for s in selected
        if window[s].count() >= cfg.vol_lookback
        and np.isfinite(window[s].std())
        and window[s].std() * np.sqrt(252) > 1e-4
    ]
    if not valid:
        return pd.Series(dtype=float), np.nan, np.nan, np.nan, (), 0.0
    window = window[valid].dropna()
    if len(window) < cfg.vol_lookback:
        return pd.Series(dtype=float), np.nan, np.nan, np.nan, (), 0.0
    inverse = 1 / (window.std(ddof=1) * np.sqrt(252)).clip(lower=1e-4)
    sleeve = min(len(valid) / cfg.top_n, 1.0)
    preliminary = inverse / inverse.sum() * sleeve
    estimated = float(
        np.sqrt(
            preliminary.to_numpy() @ window.cov().to_numpy() @ preliminary.to_numpy()
        )
        * np.sqrt(252)
    )
    scalar = (
        min(cfg.target_vol / estimated, cfg.max_gross_exposure)
        if estimated > 1e-8
        else 0.0
    )
    risky = preliminary * scalar
    flags = research["high_vol_trigger"].loc[date, risky.index].fillna(False)
    triggered = tuple(flags.index[flags]) if cfg.high_vol_adjustment_enabled else ()
    before = float(risky.sum())
    if triggered:
        risky.loc[list(triggered)] *= cfg.high_vol_weight_multiplier
    return risky, estimated, scalar, sleeve, triggered, before - float(risky.sum())


def build_order_plan(weights, budget, positions, prices, min_trade_dollars):
    """Target whole strategy-symbol holdings; leave other account positions alone."""
    held = {p.symbol: Decimal(str(p.qty)) for p in positions}
    orders = []
    for symbol, weight in weights.items():
        quantity = held.get(symbol, Decimal(0))
        if quantity < 0:
            raise ValueError(f"Unexpected short position in strategy symbol {symbol}.")
        price = float(prices[symbol])
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Missing or invalid execution price for {symbol}.")
        # Negative notebook cash denotes financing, not a short cash-ETF trade.
        target = (
            Decimal(str(max(0.0, weight) * budget)) / Decimal(str(price))
        ).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        delta = target - quantity
        notional = float(abs(delta)) * price
        if delta == 0 or (target != 0 and notional < min_trade_dollars):
            continue
        orders.append(
            {
                "symbol": symbol,
                "side": "buy" if delta > 0 else "sell",
                "qty": str(abs(delta)),
                "estimated_notional": notional,
            }
        )
    return sorted(orders, key=lambda order: order["side"] != "sell")


@contextmanager
def locked_state(path):
    """Serialize cron invocations sharing the same persistent state file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(path.read_text()) if path.exists() else {}
            yield state
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def save_state(path, state):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(state, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_single_iteration(
    api,
    equity_fraction=1.0,
    is_live_trade=False,
    force_rebalance=False,
    *,
    config=None,
    state_path=None,
    now=None,
    prices=None,
    execution_prices=None,
    feed="iex",
    min_trade_dollars=1.0,
    order_timeout=60.0,
    persist_state=None,
):
    """Preview or execute one rebalance.

    Normal runs use the previous calendar month's final session and execute once
    per calendar month, on the first invocation during an open market. Force uses
    the latest completed daily session and bypasses only the monthly gate.
    ``equity_fraction`` multiplies total account equity, NOT cash/buying power.
    ``is_live_trade`` enables submissions through the supplied Alpaca adapter.
    The caller configures whether that adapter uses a paper or live account.
    ``prices`` (adjusted history) and ``execution_prices`` (unadjusted prices) are
    optional overrides for reproducible previews/tests.
    Explicit ``persist_state=False`` observes the monthly calendar without any
    state-file writes. ``persist_state=True`` with trading disabled records a
    simulated monthly rebalance. Omitted persistence follows is_live_trade.
    """
    cfg = config or StrategyConfig()
    observe_only = persist_state is False
    # Existing direct previews remain stateless. PAPER callers opt into simulation state.
    persist_state = is_live_trade if persist_state is None else persist_state
    if is_live_trade and not persist_state:
        raise ValueError("Order submission requires persistent rebalance state.")
    validate_config(cfg)
    if not math.isfinite(equity_fraction) or not 0 < equity_fraction <= 1:
        raise ValueError("equity_fraction must be in (0, 1].")
    if not math.isfinite(min_trade_dollars) or min_trade_dollars < 0:
        raise ValueError("min_trade_dollars must be finite and nonnegative.")
    if not math.isfinite(order_timeout) or order_timeout <= 0:
        raise ValueError("order_timeout must be positive and finite.")
    timestamp = (
        pd.Timestamp.now(tz="America/New_York") if now is None else pd.Timestamp(now)
    )
    timestamp = (
        timestamp.tz_localize("America/New_York")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("America/New_York")
    )
    today = timestamp.tz_localize(None).normalize()
    month = today.strftime("%Y-%m")
    account = api.get_account()
    account_id = str(account.id)
    path = (
        Path(state_path)
        if state_path
        else Path(__file__).parent / f".rotation-state-{account_id}.json"
    )
    symbols = (*cfg.universe, cfg.cash)
    scheduled_day = monthly_rebalance_day(today)
    meta = {
        "date": today.date().isoformat(),
        "scheduled_rebalance_day": scheduled_day.date().isoformat(),
        "trade_today": False,
        "rebalance_frequency": "M",
    }

    def skipped(reason):
        return {
            "status": "skipped",
            "reason": reason,
            "month": month,
            "weights": {},
            "orders": [],
            "meta": {**meta, "reason": reason},
        }

    context = (
        locked_state(path)
        if persist_state
        else nullcontext(json.loads(path.read_text()) if path.exists() else {})
    )
    with context as state:
        if state and state.get("account_id") != account_id:
            raise ValueError("State file belongs to a different Alpaca account.")
        if state.get("pending"):
            raise RuntimeError(
                f"Unfinished rebalance in {path}; reconcile its orders before clearing pending."
            )
        if not force_rebalance and state.get("last_rebalance_month") == month:
            return skipped("already_rebalanced_this_month")
        if observe_only and not force_rebalance and today != scheduled_day:
            return skipped("not_scheduled_rebalance_day")
        if is_live_trade:
            if getattr(account, "trading_blocked", False):
                raise RuntimeError("Alpaca account is blocked from trading.")
            if not api.get_clock().is_open:
                return skipped("market_closed")
            if any(order.symbol in symbols for order in api.list_orders()):
                raise RuntimeError(
                    "Open orders exist for strategy ETFs; wait for or reconcile them first."
                )

        if prices is None:
            prices = load_prices(
                api,
                symbols,
                cfg.history_start,
                timestamp.normalize().isoformat(),
                feed=feed,
            )
        prices = normalize_prices(prices, symbols, today)
        candidates = (
            prices
            if force_rebalance
            else prices.loc[prices.index < today.replace(day=1)]
        )
        if candidates.empty:
            raise ValueError("No completed signal session available.")
        date = candidates.index[-1]
        expected_month = (today.to_period("M") - 1) if not force_rebalance else None
        if expected_month is not None and date.to_period("M") != expected_month:
            raise ValueError("Price history does not include the previous month.")
        boundary = today if force_rebalance else today.replace(day=1)
        if (boundary - date).days > 7 or prices.loc[date, list(symbols)].isna().any():
            raise ValueError("Signal prices are stale or missing for one or more ETFs.")
        needed = max(
            max(cfg.momentum_lookbacks) + 1, cfg.trend_ma, cfg.vol_lookback + 1
        )
        if any(candidates[s].count() < needed for s in cfg.universe):
            raise ValueError(
                f"Insufficient history: require at least {needed} observations per risky ETF."
            )
        research = prepare_research_data(candidates, cfg)
        positions = api.list_positions()
        previous = state.get(
            "selected",
            [
                p.symbol
                for p in positions
                if p.symbol in cfg.universe and float(p.qty) > 0
            ],
        )
        selected, ranks, scores = select_etfs(research, date, previous, cfg)
        risky, estimated_vol, scalar, sleeve, triggered, reduction = calculate_weights(
            selected, date, research, cfg
        )
        weights = {s: float(risky.get(s, 0)) for s in cfg.universe}
        weights[cfg.cash] = 1 - float(risky.sum())
        equity = float(account.equity)
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("Account equity must be positive and finite.")
        budget = equity * equity_fraction
        if execution_prices is None:
            raw = load_prices(
                api,
                symbols,
                (today - timedelta(days=10)).isoformat(),
                timestamp.normalize().isoformat(),
                adjustment="raw",
                feed=feed,
            )
            raw = normalize_prices(raw, symbols, today)
            if raw.empty or (today - raw.index[-1]).days > 7:
                raise ValueError("Execution prices are stale or missing.")
            execution_prices = raw.iloc[-1]
        plan = build_order_plan(
            weights, budget, positions, execution_prices, min_trade_dollars
        )
        result = {
            "status": "preview",
            "signal_date": date.date().isoformat(),
            "month": month,
            "portfolio_equity": equity,
            "allocation_value": budget,
            "selected": selected,
            "ranks": ranks,
            "scores": scores,
            "target_weights": weights,
            "target_values": {s: w * budget for s, w in weights.items()},
            "financing_value": max(0.0, -weights[cfg.cash] * budget),
            "high_vol_etfs": list(triggered),
            "high_vol_weight_reduction": reduction,
            "orders": plan,
        }
        if not is_live_trade:
            result["weights"] = weights
            result["meta"] = {
                **meta,
                "trade_today": True,
                "market_data_date": result["signal_date"],
                "gross_risky": float(risky.sum()),
                "cash_weight": weights[cfg.cash],
            }
            if persist_state:
                state.update(
                    account_id=account_id,
                    last_rebalance_month=month,
                    last_signal_date=result["signal_date"],
                    selected=selected,
                    pending=None,
                )
                save_state(path, state)
                result["status"] = "simulated"
            return result

        # Check estimated funding before any sales; recheck after sale fills below.
        sells = sum(o["estimated_notional"] for o in plan if o["side"] == "sell")
        buys = sum(o["estimated_notional"] for o in plan if o["side"] == "buy")
        buying_power = float(account.buying_power)
        if not math.isfinite(buying_power) or buys > buying_power + sells:
            raise RuntimeError("Insufficient buying power for the target allocation.")
        state.update(
            account_id=account_id,
            pending={"month": month, "plan": plan, "submitted": []},
        )
        save_state(path, state)
        for order in plan:
            if order["side"] == "buy":
                available = float(api.get_account().buying_power)
                if (
                    not math.isfinite(available)
                    or order["estimated_notional"] > available
                ):
                    raise RuntimeError(
                        f"Insufficient buying power for {order['symbol']}; rebalance remains pending."
                    )
            submitted = api.submit_order(
                symbol=order["symbol"],
                qty=order["qty"],
                side=order["side"],
                type="market",
                time_in_force="day",
            )
            state["pending"]["submitted"].append(str(submitted.id))
            save_state(path, state)
            deadline = time.monotonic() + order_timeout
            while True:
                current = api.get_order(str(submitted.id))
                status = getattr(current.status, "value", current.status)
                if status == "filled":
                    break
                if status in {
                    "canceled",
                    "expired",
                    "rejected",
                    "suspended",
                    "done_for_day",
                }:
                    raise RuntimeError(
                        f"Order {submitted.id} ended with {status}; rebalance remains pending."
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Order {submitted.id} not filled; rebalance remains pending."
                    )
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        state.update(
            last_rebalance_month=month,
            last_signal_date=result["signal_date"],
            selected=selected,
            pending=None,
        )
        save_state(path, state)
        result["status"] = "rebalanced"
        result["weights"] = weights
        result["meta"] = {
            **meta,
            "trade_today": True,
            "market_data_date": result["signal_date"],
            "gross_risky": float(risky.sum()),
            "cash_weight": weights[cfg.cash],
        }
        return result


def export_strategy_json(
    result: dict,
    output_path: str = "strategy_export.json",
    *,
    strategy_name: str = "etf-adaptive-rotation",
    equity_fraction: float = 1.0,
    trade_today: bool | None = None,
    liquidate_when_inactive: bool | None = None,
) -> dict:
    """
    Convert a run_single_iteration() result into the external strategy JSON shape
    and write it to disk.

    Rules:
    - positions are all non-zero target weights from result["weights"]
    - when trade_today is true, positions are exported exactly from the target portfolio
    - capital_requested uses meta["gross_risky"] when available, else gross exposure
    - gross/net exposure are computed from the exported positions
    - holding_period_days is inferred from the configured rebalance cadence
    - trade_today is inferred from the strategy result when not provided
    - liquidate_when_inactive is true when equity_fraction == 0, else false
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dict returned by run_single_iteration()")

    weights = result.get("weights") or {}
    if not isinstance(weights, dict):
        raise TypeError('result["weights"] must be a dict when present')

    positions = []
    for symbol, target_weight in weights.items():
        weight = float(target_weight)
        if abs(weight) <= 1e-12:
            continue
        positions.append({"symbol": symbol, "target_weight": weight})

    gross_exposure = float(sum(abs(p["target_weight"]) for p in positions))
    net_exposure = float(sum(p["target_weight"] for p in positions))

    meta = result.get("meta", {}) or {}
    capital_requested = float(meta.get("gross_risky", gross_exposure))
    updated_date = str(
        meta.get("date") or pd.Timestamp.now(tz="America/New_York").date()
    )

    frequency = meta.get("rebalance_frequency", "M")
    holding_period_days = 30 if frequency == "M" else 7 if frequency == "W" else 0
    if trade_today is None:
        trade_today = result_trade_today(result)
    if liquidate_when_inactive is None:
        liquidate_when_inactive = float(equity_fraction) == 0.0

    payload = {
        "strategy": strategy_name,
        "updated_date": updated_date,
        "initialize_portfolio": True,
        "trade_today": trade_today,
        "liquidate_when_inactive": liquidate_when_inactive,
        "capital_requested": capital_requested,
        "positions": positions,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "holding_period_days": holding_period_days,
    }

    with open(output_path, "w", encoding="ascii") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    return payload


def upload_file_to_digitalocean_spaces(
    file_path: str,
    *,
    bucket_name: str,
    region: str,
    object_key: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    content_type: str = "application/json",
    acl: str | None = None,
) -> dict:
    """Upload a local file to DigitalOcean Spaces via the S3-compatible API."""
    import boto3

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not os.path.isfile(file_path):
        raise IsADirectoryError(
            f"Expected a file path, got a directory or non-file path: {file_path}"
        )
    if not bucket_name:
        raise ValueError("bucket_name is required")
    if not region:
        raise ValueError("region is required")

    spaces_key = access_key or os.getenv("DO_SPACES_KEY")
    spaces_secret = secret_key or os.getenv("DO_SPACES_SECRET")
    if not spaces_key or not spaces_secret:
        raise ValueError(
            "DigitalOcean Spaces credentials are required via arguments or "
            "DO_SPACES_KEY / DO_SPACES_SECRET environment variables."
        )

    key = object_key or os.path.basename(file_path)
    endpoint_url = f"https://{region}.digitaloceanspaces.com"

    client = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=spaces_key,
        aws_secret_access_key=spaces_secret,
    )

    extra_args = {"ContentType": content_type}
    if acl:
        extra_args["ACL"] = acl

    client.upload_file(file_path, bucket_name, key, ExtraArgs=extra_args)

    return {
        "bucket": bucket_name,
        "region": region,
        "object_key": key,
        "endpoint_url": endpoint_url,
        "object_url": f"https://{bucket_name}.{region}.digitaloceanspaces.com/{key}",
    }


# ==========================================
# Optional: simple pretty printer
# ==========================================
def print_orders_table(result: dict):
    lines = []

    def emit(line=""):
        lines.append(line)
        print(line)

    def fmt_num(x, prec=3):
        return (
            f"{x:.{prec}f}"
            if isinstance(x, (int, float, np.floating)) and pd.notna(x)
            else "N/A"
        )

    meta = result.get("meta", {}) or {}
    date = meta.get("date", "N/A")
    reason = meta.get("reason")  # present when it's not a rebalance day
    on = meta.get("on_sleeves", [])
    gross = fmt_num(meta.get("gross_risky"))
    cash = fmt_num(meta.get("cash_weight"))

    # --- NEW: dynamic VT regime diagnostics (stress level) ---
    vt_mode = meta.get("vt_mode", "static")
    vt_target_used = meta.get("vt_target_used", None)
    vt_lkbk_used = meta.get("vt_lkbk_used", None)

    vt_diag = meta.get("vt_diag", {}) or {}
    stress_level = vt_diag.get("regime", "N/A")
    vol_60 = vt_diag.get("vol_60", np.nan)
    dd_6m = vt_diag.get("dd_6m", np.nan)
    avg_corr_60 = vt_diag.get("avg_corr_60", np.nan)

    if reason:
        emit(f"Rebalance date: {date} | reason={reason}")
        if not result.get("orders"):
            emit("(No orders)")
        return "\n".join(lines) + "\n"

    header = (
        f"Rebalance date: {date} | sleeves_on={on} | gross_sleeves={gross} | cash={cash}\n"
        f"VT mode={vt_mode} | stress_level={stress_level} | "
        f"vt_target_used={fmt_num(vt_target_used, 3)} | vt_lkbk_used={vt_lkbk_used}\n"
        f"diag: vol_60={fmt_num(vol_60, 3)} | dd_6m={fmt_num(dd_6m, 3)} | avg_corr_60={fmt_num(avg_corr_60, 3)}"
    )
    emit(header)

    orders = result.get("orders", [])
    if not orders:
        emit("(No orders)")
        return "\n".join(lines) + "\n"

    emit("symbol   action   target_qty   diff     price     alloc_w")
    for o in orders:
        px = o.get("price", np.nan)
        alloc_w = o.get("alloc_w", np.nan)
        emit(
            f"{o['symbol']:6}  {o['action']:6}  {int(o['target_qty']):11d}  {int(o['diff']):6d}  "
            f"{(float(px) if pd.notna(px) else np.nan):9.4f}  {float(alloc_w):.4f}"
        )

    return "\n".join(lines) + "\n"


def print_weights_table(result: dict):
    lines = []

    def emit(line=""):
        lines.append(line)
        print(line)

    meta = result.get("meta", {}) or {}
    weights = result.get("weights", {}) or {}
    date = meta.get("date", "N/A")
    asof = meta.get("asof", "N/A")
    reason = meta.get("reason")

    if reason:
        emit(f"Observation date: {date} | reason={reason}")
        return "\n".join(lines) + "\n"

    emit(f"Observation date: {date} | asof={asof}")
    emit("symbol   alloc_pct")

    non_zero = [
        (symbol, float(weight))
        for symbol, weight in weights.items()
        if abs(float(weight)) > 1e-12
    ]

    for symbol, weight in sorted(non_zero, key=lambda item: item[1], reverse=True):
        emit(f"{symbol:6}  {weight * 100:8.2f}%")

    if not non_zero:
        emit("(No active allocations)")

    return "\n".join(lines) + "\n"
