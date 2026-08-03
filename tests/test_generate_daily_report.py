import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_daily_report import compact_messages, empty_report, extract_output_text


class GenerateDailyReportTests(unittest.TestCase):
    def test_extract_output_text(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ]
        }
        self.assertEqual(extract_output_text(response), "hello")

    def test_compact_messages_keeps_verification_link(self) -> None:
        payload = {
            "messages": [
                {
                    "timestamp": "2026-08-02T01:02:03+08:00",
                    "channel_name": "research",
                    "author": {"username": "dev"},
                    "content": "A technical proposal",
                    "url": "https://discord.com/channels/1/2/3",
                    "attachments": [],
                }
            ]
        }
        text, included = compact_messages(payload, 10_000)
        self.assertEqual(included, 1)
        self.assertIn("https://discord.com/channels/1/2/3", text)

    def test_empty_report_has_required_sections(self) -> None:
        report = empty_report({})
        for title in (
            "## 总览 Brief",
            "## RWKV 技术相关讨论",
            "## Bug 与问题",
            "## 社区反馈",
            "## General",
        ):
            self.assertIn(title, report)


if __name__ == "__main__":
    unittest.main()
