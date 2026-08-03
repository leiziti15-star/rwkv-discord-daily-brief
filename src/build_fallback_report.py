#!/usr/bin/env python3
"""Build a deterministic fallback report from collected Discord messages.

The scheduled Codex task can replace this with a higher-quality semantic summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TECH_TERMS = (
    "rwkv",
    "architecture",
    "attention",
    "token",
    "inference",
    "training",
    "finetune",
    "fine-tune",
    "cuda",
    "vulkan",
    "quant",
    "模型",
    "架构",
    "推理",
    "训练",
    "微调",
    "量化",
    "显存",
    "性能",
)
BUG_TERMS = (
    "bug",
    "error",
    "exception",
    "fail",
    "failed",
    "crash",
    "issue",
    "traceback",
    "报错",
    "失败",
    "崩溃",
    "问题",
    "异常",
    "无法",
)
COMMUNITY_TERMS = (
    "feedback",
    "request",
    "feature",
    "document",
    "docs",
    "tutorial",
    "guide",
    "example",
    "反馈",
    "需求",
    "建议",
    "文档",
    "教程",
    "指南",
    "示例",
)
QUESTION_TERMS = ("?", "？", "how ", "why ", "what ", "怎么", "为什么", "如何", "请问")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成可用的本地回退日报")
    parser.add_argument("input", help="discord_messages_YYYY-MM-DD.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-items", type=int, default=8)
    return parser.parse_args()


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def classify(message: dict[str, Any]) -> str:
    context = " ".join(
        filter(
            None,
            [
                message.get("category_name"),
                message.get("parent_name"),
                message.get("channel_name"),
                message.get("content"),
            ],
        )
    )
    if contains_any(context, BUG_TERMS):
        return "bug"
    if contains_any(context, TECH_TERMS):
        return "tech"
    if contains_any(context, COMMUNITY_TERMS):
        return "community"
    return "general"


def content_for(message: dict[str, Any]) -> str:
    content = re.sub(r"\s+", " ", message.get("content", "")).strip()
    if not content and message.get("attachments"):
        content = "附件：" + "、".join(
            item.get("filename") or "未命名附件"
            for item in message["attachments"]
        )
    if not content and message.get("embeds"):
        content = "；".join(
            item.get("title") or item.get("description") or "外部链接"
            for item in message["embeds"]
        )
    return content[:260]


def priority(message: dict[str, Any]) -> int:
    text = message.get("content", "")
    score = sum(item.get("count", 0) for item in message.get("reactions", []))
    if contains_any(text, QUESTION_TERMS):
        score += 5
    if contains_any(text, BUG_TERMS):
        score += 4
    if contains_any(text, COMMUNITY_TERMS):
        score += 3
    return score


def item_line(message: dict[str, Any]) -> str:
    channel = message.get("channel_name") or message["channel_id"]
    author = message.get("author", {}).get("username") or "unknown"
    content = content_for(message)
    return f"- **#{channel} · {author}**：{content} ([原消息]({message['url']}))"


def build_report(payload: dict[str, Any], max_items: int) -> str:
    messages = payload.get("messages", [])
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        buckets[classify(message)].append(message)
    for items in buckets.values():
        items.sort(key=lambda item: (priority(item), item["timestamp"]), reverse=True)

    channel_counts = Counter(item.get("channel_name", "unknown") for item in messages)
    top_channels = "、".join(
        f"#{name}（{count}）" for name, count in channel_counts.most_common(5)
    ) or "无"
    actions = sorted(
        (
            item
            for item in messages
            if contains_any(item.get("content", ""), QUESTION_TERMS)
            or contains_any(item.get("content", ""), COMMUNITY_TERMS)
            or contains_any(item.get("content", ""), BUG_TERMS)
        ),
        key=priority,
        reverse=True,
    )[:3]

    lines = [
        f"# RWKV Discord Daily Brief | {payload['report_date']}",
        "",
        "## 总览 Brief",
        "",
        (
            f"- 昨日共有 **{payload['stats']['message_count']}** 条有效消息，"
            f"涉及 **{payload['stats']['active_channel_count']}** 个频道、"
            f"**{payload['stats']['active_user_count']}** 位参与者。"
        ),
        f"- 最活跃频道：{top_channels}",
    ]
    if actions:
        lines.extend(["- **建议作者优先关注：**"])
        lines.extend(
            f"  {item_line(item)[2:]}" for item in actions
        )
    else:
        lines.append("- 今日没有识别出需要作者立即回复的问题。")

    sections = [
        ("tech", "RWKV 技术相关讨论"),
        ("bug", "Bug 与问题"),
        ("community", "社区反馈"),
        ("general", "General"),
    ]
    for key, title in sections:
        lines.extend(["", f"## {title}", ""])
        items = buckets.get(key, [])[:max_items]
        if not items:
            lines.append("- 无值得报告的新内容。")
        else:
            lines.extend(item_line(item) for item in items)

    if payload.get("warnings"):
        lines.extend(
            [
                "",
                "> 采集提示：有部分频道无法读取，详见原始 JSON 的 `warnings` 字段。",
            ]
        )
    lines.extend(
        [
            "",
            "---",
            "说明：这是一份本地回退版摘要；正式定时任务会由 Codex 做语义归并、去重和优先级判断。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"RWKV_Daily_Brief_{payload['report_date']}.md"
    output_path.write_text(build_report(payload, args.max_items), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

