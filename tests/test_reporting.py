from contextlib import ExitStack, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import algo


class ReportingTests(unittest.TestCase):
    def test_modes_email_and_spaces_wiring(self):
        for mode in ("LIVE", "PAPER", "OBSERVE"):
            with self.subTest(mode=mode), ExitStack() as stack:
                stack.enter_context(patch.dict("os.environ", {
                    "APP_STATE": mode, "EMAIL_POSITIONS": "true",
                    "FROM_ADDRESS": "sender@example.com",
                    "TO_ADDRESSES": "one@example.com, two@example.com",
                    "SYNC_STRATEGY_JSON_TO_SPACES": "true",
                    "SPACES_OBJECT_KEY_PATH": "/strategies/",
                }, clear=True))
                stack.enter_context(patch("sys.argv", ["algo.py"]))
                stack.enter_context(patch("algo.load_dotenv"))
                stack.enter_context(redirect_stdout(io.StringIO()))
                factory = stack.enter_context(patch("algo.AlpacaAPI.from_env"))
                factory.return_value.get_account.return_value = SimpleNamespace(id="test", equity="100000")
                run = stack.enter_context(patch("algo.run_single_iteration", return_value={
                    "status": "skipped", "weights": {}, "orders": [],
                    "meta": {"date": "2026-09-21", "trade_today": False,
                             "reason": "not_scheduled_rebalance_day"},
                }))
                snapshot = stack.enter_context(patch("algo.build_strategy_snapshot_for_reporting", return_value={
                    "weights": {"QQQ": 0.8, "BIL": 0.2}, "orders": [],
                }))
                export = stack.enter_context(patch("algo.export_strategy_json"))
                upload = stack.enter_context(patch("algo.upload_file_to_digitalocean_spaces"))
                ses = stack.enter_context(patch("algo.AmazonSES"))
                algo.main()
                self.assertEqual(run.call_args.kwargs["is_live_trade"], mode == "LIVE")
                self.assertEqual(run.call_args.kwargs["persist_state"], mode != "OBSERVE")
                self.assertEqual(snapshot.call_count, int(mode == "OBSERVE"))
                self.assertEqual(export.call_args.kwargs["output_path"], "etf-adaptive-rotation.json")
                self.assertFalse(export.call_args.kwargs["trade_today"])
                self.assertEqual(upload.call_args.kwargs["object_key"], "strategies/etf-adaptive-rotation.json")
                self.assertEqual(ses.return_value.send_html_email.call_count, 2)

    def test_invalid_mode_fails_before_connecting(self):
        with patch.dict("os.environ", {"APP_STATE": "typo"}, clear=True), \
                patch("algo.load_dotenv"), patch("sys.argv", ["algo.py"]), \
                patch("algo.AlpacaAPI.from_env") as factory:
            with self.assertRaisesRegex(ValueError, "APP_STATE"):
                algo.main()
            factory.assert_not_called()
