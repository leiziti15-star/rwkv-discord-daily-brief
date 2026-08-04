import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_daily_report import (
    api_endpoint,
    build_request_body,
    compact_messages,
    empty_report,
    extract_chat_completion_stream,
    extract_chat_completion_text,
    extract_output_text,
    normalize_report,
    validate_report,
)


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

    def test_extract_chat_completion_text(self) -> None:
        response = {"choices": [{"message": {"content": "hello from chat"}}]}
        self.assertEqual(extract_chat_completion_text(response), "hello from chat")

    def test_extract_chat_completion_stream(self) -> None:
        response = [
            b'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
            b'data: {"choices":[{"delta":{"content":"stream"}}]}\n',
            b'data: [DONE]\n',
        ]
        self.assertEqual(extract_chat_completion_stream(response), "hello stream")

    def test_api_endpoint(self) -> None:
        self.assertEqual(
            api_endpoint("https://example.com/v1", "responses"),
            "https://example.com/v1/responses",
        )
        self.assertEqual(
            api_endpoint("https://example.com/v1", "chat_completions"),
            "https://example.com/v1/chat/completions",
        )

    def test_api_endpoint_requires_https(self) -> None:
        with self.assertRaises(ValueError):
            api_endpoint("http://example.com/v1", "responses")

    def test_chat_request_body(self) -> None:
        body = build_request_body("model-x", "chat_completions", "prompt")
        self.assertEqual(body["model"], "model-x")
        self.assertEqual(body["messages"][-1]["content"], "prompt")
        self.assertTrue(body["stream"])

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

    def test_validate_report_accepts_complete_report(self) -> None:
        validate_report(empty_report({}))

    def test_validate_report_rejects_truncated_source_link(self) -> None:
        report = empty_report({}) + "\n[原消息](https://discord.com/channels/1/2"
        with self.assertRaises(RuntimeError):
            validate_report(report)

    def test_normalize_report_drops_truncated_line_and_restores_sections(self) -> None:
        report = """## 总览 Brief

- 可验证要点。[原消息](https://discord.com/channels/1/2/3)

## RWKV 技术相关讨论

- 被截断的要点。[原消息](https://discord.com/channels/1/2
"""
        normalized = normalize_report(report)
        validate_report(normalized)
        self.assertIn("可验证要点", normalized)
        self.assertNotIn("被截断的要点", normalized)
        self.assertIn("## General\n\n- 无值得报告的新内容。", normalized)

    def test_normalize_report_drops_source_less_claims(self) -> None:
        report = empty_report({}).replace(
            "- 无值得报告的新内容。",
            "- 没有来源的概括。",
            1,
        )
        normalized = normalize_report(report)
        self.assertNotIn("没有来源的概括", normalized)
        self.assertIn("## RWKV 技术相关讨论\n\n- 无值得报告的新内容。", normalized)


if __name__ == "__main__":
    unittest.main()
