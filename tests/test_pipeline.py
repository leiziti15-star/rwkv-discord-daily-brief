from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from build_fallback_report import build_report, classify
from discord_daily import choose_guild, datetime_to_snowflake, report_window
from send_wecom import split_markdown, utf8_len


class DailyBriefTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = ROOT / "tests" / "fixtures" / "sample_messages.json"
        self.payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))

    def test_classification(self) -> None:
        self.assertEqual(classify(self.payload["messages"][0]), "tech")
        self.assertEqual(classify(self.payload["messages"][1]), "bug")
        self.assertEqual(classify(self.payload["messages"][2]), "community")
        self.assertEqual(classify(self.payload["messages"][3]), "general")

    def test_report_sections_and_links(self) -> None:
        report = build_report(self.payload, max_items=8)
        self.assertIn("## 总览 Brief", report)
        self.assertIn("## RWKV 技术相关讨论", report)
        self.assertIn("## Bug 与问题", report)
        self.assertIn("## 社区反馈", report)
        self.assertIn("## General", report)
        self.assertIn("https://discord.com/channels/", report)

    def test_wecom_chunking_stays_under_limit(self) -> None:
        text = ("中文内容 abc\n" * 1000).strip()
        chunks = split_markdown(text, limit=1000)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(utf8_len(chunk) <= 1000 for chunk in chunks))

    def test_report_window(self) -> None:
        window = report_window("2026-07-30", "Asia/Shanghai")
        self.assertEqual(window.start_utc.isoformat(), "2026-07-29T16:00:00+00:00")
        self.assertLess(
            datetime_to_snowflake(window.start_utc),
            datetime_to_snowflake(window.end_utc),
        )

    def test_choose_guild_by_name(self) -> None:
        guilds = [
            {"id": "1", "name": "Sandbox"},
            {"id": "2", "name": "RWKV Community"},
        ]
        self.assertEqual(choose_guild(guilds, "", "RWKV")["id"], "2")

    def test_choose_only_guild(self) -> None:
        self.assertEqual(
            choose_guild([{"id": "9", "name": "Only Server"}], "", "")["id"],
            "9",
        )


if __name__ == "__main__":
    unittest.main()
