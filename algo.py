"""Configure Alpaca and run one monthly ETF rotation iteration."""

import argparse
import json
import os
from html import escape
from pathlib import Path

import pandas as pd
from alpaca_adapter import AlpacaAPI
from dotenv import load_dotenv
from helpers import (
    StrategyConfig,
    build_strategy_snapshot_for_reporting,
    str2bool,
    getenv_float,
    getenv_int,
    export_strategy_json,
    format_observe_allocations,
    get_app_state,
    is_paper_account,
    monthly_rebalance_day,
    result_trade_today,
    run_single_iteration,
    upload_file_to_digitalocean_spaces,
    validate_config,
)
from SES import AmazonSES


def strategy_config_from_env():
    """Read strategy settings after .env has been loaded."""
    config = StrategyConfig(
        universe=tuple(
            s.strip().upper()
            for s in os.getenv("ETF_UNIVERSE", "QQQ,EFA,TLT,GLD,VNQ").split(",")
        ),
        cash=os.getenv("CASH_ETF", "BIL").strip().upper(),
        history_start=os.getenv("HISTORY_START", "2006-01-01"),
        top_n=getenv_int("TOP_N", 3),
        exit_rank=getenv_int("EXIT_RANK", 5),
        momentum_lookbacks=tuple(
            int(n.strip())
            for n in os.getenv("MOMENTUM_LOOKBACKS", "63,126,252").split(",")
        ),
        momentum_weights=tuple(
            float(w.strip())
            for w in os.getenv("MOMENTUM_WEIGHTS", "0.50,0.30,0.20").split(",")
        ),
        trend_ma=getenv_int("TREND_MA", 200),
        vol_lookback=getenv_int("VOL_LOOKBACK", 20),
        target_vol=getenv_float("TARGET_VOL", 0.20),
        max_gross_exposure=getenv_float("MAX_GROSS_EXPOSURE", 1.50),
        high_vol_adjustment_enabled=str2bool(os.getenv("HIGH_VOL_ADJUSTMENT_ENABLED", True)),
        high_vol_threshold=getenv_float("HIGH_VOL_THRESHOLD", 1.20),
        high_vol_weight_multiplier=getenv_float("HIGH_VOL_WEIGHT_MULTIPLIER", 0.70),
        high_vol_reference_lookback=getenv_int("HIGH_VOL_REFERENCE_LOOKBACK", 252),
        use_ema_smoothing=str2bool(os.getenv("USE_EMA_SMOOTHING", True)),
        ema_alpha=getenv_float("EMA_ALPHA", 0.30),
        use_hysteresis=str2bool(os.getenv("USE_HYSTERESIS", True)),
    )
    if not config.cash or any(not symbol for symbol in config.universe):
        raise ValueError("ETF_UNIVERSE and CASH_ETF must not contain empty symbols.")
    validate_config(config)
    return config


def main():
    load_dotenv()
    config = strategy_config_from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equity-fraction",
        type=float,
        default=getenv_float("EQUITY_FRACTION", 1.0),
    )
    parser.add_argument(
        "--force-rebalance",
        action=argparse.BooleanOptionalAction,
        default=str2bool(os.getenv("FORCED_REBALANCE", os.getenv("FORCE_REBALANCE", False))),
    )
    parser.add_argument("--state-path", default=os.getenv("STATE_PATH"))
    args = parser.parse_args()
    app_state = get_app_state()
    is_observe = app_state == "OBSERVE"
    sync_strategy_json_to_spaces = str2bool(os.getenv("SYNC_STRATEGY_JSON_TO_SPACES", False))
    email_positions = str2bool(os.getenv("EMAIL_POSITIONS", False))
    to_addresses = [
        a.strip() for a in os.getenv("TO_ADDRESSES", "").split(",") if a.strip()
    ]
    if email_positions and (not to_addresses or not os.getenv("FROM_ADDRESS")):
        raise ValueError("EMAIL_POSITIONS requires FROM_ADDRESS and TO_ADDRESSES.")
    print(f"Running in {app_state} mode")

    api = AlpacaAPI.from_env(
        api_key=os.getenv("ALPACA_KEY_ID"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=is_paper_account(),
    )

    account = api.get_account()
    portfolio_value = round(float(account.equity), 3)
    print(f"Portfolio value: {portfolio_value:.3f}")
    # Simulated PAPER completion must not suppress a subsequent real rebalance.
    state_path = args.state_path
    if app_state == "PAPER" and state_path is None:
        state_path = str(
            Path(__file__).parent / f".rotation-state-{account.id}-simulation.json"
        )
    result = run_single_iteration(
        api=api,
        config=config,
        equity_fraction=args.equity_fraction,
        is_live_trade=app_state == "LIVE",
        force_rebalance=args.force_rebalance,
        state_path=state_path,
        persist_state=not is_observe,
    )
    scheduled_trade_today = result_trade_today(result)
    reporting_portfolio = result
    if is_observe and not scheduled_trade_today:
        reporting_portfolio = build_strategy_snapshot_for_reporting(
            result,
            api=api,
            config=config,
            equity_fraction=args.equity_fraction,
            state_path=state_path,
        )

    if sync_strategy_json_to_spaces:
        output_path = "etf-adaptive-rotation.json"
        export_strategy_json(
            result=reporting_portfolio,
            output_path=output_path,
            strategy_name="etf-adaptive-rotation",
            equity_fraction=args.equity_fraction,
            trade_today=scheduled_trade_today,
            liquidate_when_inactive=False if is_observe else None,
        )
        prefix = os.getenv("SPACES_OBJECT_KEY_PATH", "").strip("/")
        upload_file_to_digitalocean_spaces(
            file_path=output_path,
            region=os.getenv("SPACES_REGION"),
            object_key=f"{prefix}/{output_path}" if prefix else output_path,
            bucket_name=os.getenv("SPACES_BUCKET"),
            access_key=os.getenv("SPACES_KEY"),
            secret_key=os.getenv("SPACES_SECRET"),
        )

    meta = result.get("meta", {})
    date = pd.Timestamp(
        meta.get("date") or pd.Timestamp.now(tz="America/New_York").date()
    )
    scheduled = monthly_rebalance_day(date)
    next_day = (
        scheduled
        if date < scheduled
        else monthly_rebalance_day(date + pd.offsets.MonthBegin(1))
    )
    lines = [
        f"Adaptive Rotation Report - {app_state.title()}",
        f"Portfolio Value: ${portfolio_value:,.2f}",
        f"Trade Today: {'Yes' if scheduled_trade_today else 'No'}",
        f"Next Rebalance Day: {next_day.date()}",
        f"Status: {result.get('status', 'unknown')}",
    ]
    if meta.get("reason"):
        lines.append(f"Reason: {meta['reason']}")
    lines.extend(
        [
            "",
            "Portfolio Allocations:",
            *format_observe_allocations(reporting_portfolio.get("weights", {})),
        ]
    )
    if not is_observe:
        lines.extend(["", "Orders (simulated):" if app_state == "PAPER" else "Orders:"])
        lines.extend(
            f"{o['symbol']}: {o['side']} {o['qty']} shares"
            for o in result.get("orders", [])
        )
    message_body_plain = "\n".join(lines)
    if email_positions:
        ses = AmazonSES(
            region=os.getenv("AWS_SES_REGION_NAME"),
            access_key=os.getenv("AWS_SES_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SES_SECRET_ACCESS_KEY"),
            from_address=os.getenv("FROM_ADDRESS"),
        )
        for to_address in to_addresses:
            ses.send_html_email(
                to_address=to_address,
                subject=f"Adaptive Rotation Algo Report - {app_state.title()}",
                content=f"<pre>{escape(message_body_plain)}</pre>",
            )
    print(message_body_plain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
