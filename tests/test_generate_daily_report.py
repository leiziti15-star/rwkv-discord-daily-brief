import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_daily_report import (
    INSTRUCTIONS,
    add_activity_summary,
    add_todo_summary,
    api_endpoint,
    build_prompt,
    build_request_body,
    compact_messages,
    empty_report,
    enforce_todo_window,
    extract_chat_completion_stream,
    extract_chat_completion_text,
    extract_output_text,
    load_recent_todo_reminders,
    normalize_report,
    recent_todo_items,
    replace_todo_items,
    unwrap_source_link_code,
    validate_report,
)


class GenerateDailyReportTests(unittest.TestCase):
    def test_instructions_apply_rwkv_relevance_gate_to_august_4_cases(self) -> None:
        self.assertIn("未关联 RWKV 的 llama.cpp/Qwen 内核问题", INSTRUCTIONS)
        self.assertIn("DeepSeek 观点争议", INSTRUCTIONS)
        self.assertIn("媒体内容质量评价应过滤", INSTRUCTIONS)
        self.assertIn("只能写入 General", INSTRUCTIONS)

    def test_instructions_keep_official_posts_separate_and_todo_last(self) -> None:
        self.assertIn("不要仅因发言者是 BlinkDL 就视为官方发布", INSTRUCTIONS)
        self.assertLess(
            INSTRUCTIONS.index("## RWKV 官方发布与作者动态"),
            INSTRUCTIONS.index("## RWKV 技术相关讨论"),
        )
        self.assertLess(
            INSTRUCTIONS.index("## General"),
            INSTRUCTIONS.index("## RWKV 待跟进问题"),
        )

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
            "## RWKV 官方发布与作者动态",
            "## RWKV 技术相关讨论",
            "## Bug 与问题",
            "## 社区反馈",
            "## General",
            "## RWKV 待跟进问题",
        ):
            self.assertIn(title, report)

    def test_validate_report_accepts_complete_report(self) -> None:
        validate_report(empty_report({}))

    def test_validate_report_rejects_truncated_source_link(self) -> None:
        report = empty_report({}) + "\n[原消息](https://discord.com/channels/1/2"
        with self.assertRaises(RuntimeError):
            validate_report(report)

    def test_validate_report_rejects_source_link_formatted_as_code(self) -> None:
        report = empty_report({}).replace(
            "- 无值得报告的新内容。",
            "- 内容。`[原消息](https://discord.com/channels/1/2/3)`",
            1,
        )
        with self.assertRaises(RuntimeError):
            validate_report(report)

    def test_unwrap_source_link_code_hides_raw_discord_url(self) -> None:
        coded = (
            "内容。`[原消息](https://discord.com/channels/1/2/3)` "
            "``[原消息](https://discord.com/channels/1/2/4)``"
        )
        normalized = unwrap_source_link_code(coded)
        self.assertEqual(
            normalized,
            "内容。[原消息](https://discord.com/channels/1/2/3) "
            "[原消息](https://discord.com/channels/1/2/4)",
        )
        self.assertNotIn("`[原消息]", normalized)

    def test_normalize_report_unwraps_source_link_code(self) -> None:
        report = empty_report({}).replace(
            "- 无值得报告的新内容。",
            "- 内容。`[原消息](https://discord.com/channels/1/2/3)`",
            1,
        )
        normalized = normalize_report(report)
        self.assertIn("内容。[原消息](https://discord.com/channels/1/2/3)", normalized)
        self.assertNotIn("`[原消息]", normalized)
        validate_report(normalized)

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

    def test_normalize_report_keeps_structural_group_labels(self) -> None:
        report = empty_report({}).replace(
            "- 无值得报告的新内容。",
            "- **开发者需求**：\n  - 需要示例。[原消息](https://discord.com/channels/1/2/3)",
            1,
        )
        normalized = normalize_report(report)
        self.assertIn("- **开发者需求**：", normalized)
        self.assertIn("需要示例", normalized)

    def test_add_activity_summary_precedes_overview_body(self) -> None:
        payload = {
            "report_date": "2026-08-03",
            "messages": [
                {"channel_id": "1", "channel_name": "research"},
                {"channel_id": "1", "channel_name": "research"},
                {"channel_id": "2", "channel_name": "general"},
            ],
        }
        report = add_activity_summary(empty_report(payload), payload)
        expected = (
            "## 总览 Brief\n\n"
            "> **过去 24 小时频道更新**：北京时间 2026-08-03 00:00–24:00，"
            "共 **2 个频道**有更新：#research、#general。"
        )
        self.assertIn(expected, report)
        self.assertLess(report.index("过去 24 小时频道更新"), report.index("报告窗口内"))

    def test_todo_window_keeps_seven_report_dates_and_adds_rule(self) -> None:
        report = replace_todo_items(
            empty_report({}),
            [
                "- [ ] **待回复**｜提问时间：2026-08-01 09:30（北京时间）｜仍有效。[原消息](https://discord.com/channels/1/2/3)",
                "- [ ] **待回复**｜提问时间：2026-07-31 09:30（北京时间）｜已过期。[原消息](https://discord.com/channels/1/2/4)",
                "- [ ] **待回复**｜没有时间｜格式错误。[原消息](https://discord.com/channels/1/2/5)",
            ],
        )
        report = enforce_todo_window(report, "2026-08-07")
        self.assertIn("提问提醒最多保留 **7 天**，请及时回应", report)
        self.assertIn("仍有效", report)
        self.assertNotIn("已过期", report)
        self.assertNotIn("格式错误", report)

    def test_todo_summary_counts_new_and_rolling_items(self) -> None:
        report = replace_todo_items(
            empty_report({}),
            [
                "- [ ] **待回复**｜提问时间：2026-08-07 09:30（北京时间）｜新问题。[原消息](https://discord.com/channels/1/2/3)",
                "- [ ] **需文档跟进**｜提问时间：2026-08-03 10:00（北京时间）｜旧问题。[原消息](https://discord.com/channels/1/2/4)",
            ],
        )
        report = add_todo_summary(report, "2026-08-07")
        self.assertIn("本期新增 **1 项**，7 日内累计 **2 项**", report)

    def test_load_recent_todo_reminders_deduplicates_by_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            older = replace_todo_items(
                empty_report({}),
                [
                    "- [ ] **待回复**｜提问时间：2026-08-03 10:00（北京时间）｜旧摘要。[原消息](https://discord.com/channels/1/2/3)"
                ],
            )
            newer = older.replace("旧摘要", "更新摘要")
            (output_dir / "2026-08-04.md").write_text(older, encoding="utf-8")
            (output_dir / "2026-08-05.md").write_text(newer, encoding="utf-8")
            reminders = load_recent_todo_reminders(output_dir, "2026-08-06")
        self.assertEqual(len(reminders), 1)
        self.assertIn("更新摘要", reminders[0])

    def test_prompt_contains_previous_reminders(self) -> None:
        reminder = "- [ ] **待回复**｜提问时间：2026-08-03 10:00（北京时间）｜问题。[原消息](https://discord.com/channels/1/2/3)"
        prompt = build_prompt(
            {"report_date": "2026-08-04", "messages": [], "stats": {}},
            "messages",
            0,
            previous_reminders=[reminder],
        )
        self.assertIn(reminder, prompt)

    def test_recent_todo_items_ignores_future_dates(self) -> None:
        report = replace_todo_items(
            empty_report({}),
            [
                "- [ ] **待回复**｜提问时间：2026-08-08 10:00（北京时间）｜未来问题。[原消息](https://discord.com/channels/1/2/3)"
            ],
        )
        self.assertEqual(recent_todo_items(report, "2026-08-07"), [])


if __name__ == "__main__":
    unittest.main()

