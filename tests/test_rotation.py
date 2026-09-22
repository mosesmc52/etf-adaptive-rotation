import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from helpers import (StrategyConfig, calculate_weights, is_paper_account,
                     prepare_research_data, run_single_iteration, export_strategy_json)


class FakeAPI:
    def __init__(self):
        self.orders = []
        self.status = "filled"

    def get_account(self):
        return SimpleNamespace(id="test", equity="100000", cash="1000",
                               buying_power="200000", trading_blocked=False)

    def list_positions(self):
        return [SimpleNamespace(symbol="BIL", qty="100"),
                SimpleNamespace(symbol="OTHER", qty="25")]

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    def list_orders(self):
        return []

    def submit_order(self, **order):
        self.orders.append(order)
        return SimpleNamespace(id=str(len(self.orders)))

    def get_order(self, order_id):
        return SimpleNamespace(status=self.status)


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "state.json"
        self.cfg = StrategyConfig()
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2023-01-01", "2026-09-21")
        self.prices = pd.DataFrame(100 * np.exp(np.cumsum(
            rng.normal(0.001, 0.01, (len(dates), 6)), axis=0)),
            index=dates, columns=[*self.cfg.universe, self.cfg.cash])
        self.api = FakeAPI()

    def run_iteration(self, **kwargs):
        return run_single_iteration(self.api, prices=self.prices,
                                    execution_prices=self.prices.iloc[-2],
                                    state_path=self.state, now="2026-09-21 10:00", **kwargs)

    def test_preview_uses_total_equity_and_completed_month(self):
        result = self.run_iteration(equity_fraction=0.25)
        self.assertEqual(result["allocation_value"], 25000)
        self.assertEqual(result["signal_date"], "2026-08-31")
        self.assertAlmostEqual(sum(result["target_values"].values()), 25000)
        self.assertEqual(self.api.orders, [])
        self.assertFalse(self.state.exists())
        self.assertNotIn("OTHER", [o["symbol"] for o in result["orders"]])

    def test_monthly_gate_and_force(self):
        self.assertEqual(self.run_iteration(is_live_trade=True)["status"], "rebalanced")
        count = len(self.api.orders)
        self.assertEqual(self.run_iteration(is_live_trade=True)["reason"], "already_rebalanced_this_month")
        self.assertEqual(len(self.api.orders), count)
        forced = self.run_iteration(force_rebalance=True)
        self.assertEqual(forced["signal_date"], "2026-09-18")
        sides = [o["side"] for o in self.api.orders]
        self.assertEqual(sides, sorted(sides, key=lambda s: s != "sell"))

    def test_failed_order_blocks_repeat_even_when_forced(self):
        self.api.status = "rejected"
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.run_iteration(is_live_trade=True)
        with self.assertRaisesRegex(RuntimeError, "Unfinished"):
            self.run_iteration(is_live_trade=True, force_rebalance=True)
        self.assertEqual(len(self.api.orders), 1)
        self.assertNotIn("last_rebalance_month", json.loads(self.state.read_text()))

    def test_closed_market_does_not_submit(self):
        self.api.get_clock = lambda: SimpleNamespace(is_open=False)
        self.assertEqual(self.run_iteration(is_live_trade=True)["reason"], "market_closed")
        self.assertEqual(self.api.orders, [])

    def test_paper_account_setting(self):
        with patch.dict("os.environ", {"ALPACA_PAPER": "true"}):
            self.assertTrue(is_paper_account())
        with patch.dict("os.environ", {"ALPACA_PAPER": "false"}):
            self.assertFalse(is_paper_account())

    def test_observe_cadence_and_snapshot_do_not_write_state(self):
        result = self.run_iteration(persist_state=False)
        self.assertEqual(result["reason"], "not_scheduled_rebalance_day")
        self.assertFalse(result["meta"]["trade_today"])
        snapshot = self.run_iteration(persist_state=False, force_rebalance=True)
        self.assertTrue(snapshot["weights"])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertEqual(self.api.orders, [])

    def test_paper_simulation_persists_without_orders(self):
        result = self.run_iteration(is_live_trade=False, persist_state=True)
        self.assertEqual(result["status"], "simulated")
        self.assertTrue(result["meta"]["trade_today"])
        self.assertEqual(self.api.orders, [])
        self.assertEqual(self.run_iteration(persist_state=True)["reason"], "already_rebalanced_this_month")

    def test_export_matches_rotation_weights(self):
        result = self.run_iteration()
        path = str(Path(self.temp.name) / "export.json")
        payload = export_strategy_json(result, output_path=path, trade_today=False)
        self.assertEqual(payload["holding_period_days"], 30)
        self.assertEqual(payload["strategy"], "etf-adaptive-rotation")
        self.assertFalse(payload["trade_today"])
        self.assertEqual({p["symbol"]: p["target_weight"] for p in payload["positions"]},
                         {s: w for s, w in result["weights"].items() if abs(w) > 1e-12})
        self.assertEqual(json.loads(Path(path).read_text()), payload)

    def test_notebook_calculation_parity(self):
        notebook = Path(__file__).resolve().parents[1] / "backtest/strategy/diversified_etf_rotation_backtest.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        names = {"calculate_momentum", "calculate_trend_filter", "calculate_volatility",
                 "prepare_research_data", "apply_high_vol_adjustment", "calculate_weights"}
        definitions = []
        for cell in cells:
            if cell["cell_type"] == "code":
                definitions.extend(node for node in ast.parse("".join(cell["source"])).body
                                   if isinstance(node, ast.FunctionDef) and node.name in names)
        namespace = {"np": np, "pd": pd}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), "exec"), namespace)
        expected = namespace["prepare_research_data"](self.prices, self.cfg)
        actual = prepare_research_data(self.prices, self.cfg)
        for key in actual:
            pd.testing.assert_frame_equal(actual[key], expected[key])
        for selected in [[], ["QQQ"], ["QQQ", "EFA", "GLD"]]:
            for date in self.prices.index[-60::10]:
                a = calculate_weights(selected, date, actual, self.cfg)
                b = namespace["calculate_weights"](selected, date, expected, self.cfg)
                pd.testing.assert_series_equal(a[0], b[0])
                np.testing.assert_allclose(a[1:4], b[1:4], equal_nan=True)
                self.assertEqual(a[4], b[4])
                self.assertAlmostEqual(a[5], b[5])
