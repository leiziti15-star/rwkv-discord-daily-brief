#!/usr/bin/env python3
"""Send a Markdown report to an Enterprise WeChat group robot."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from discord_daily import load_dotenv


MAX_CHUNK_BYTES = 3500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="发送日报到企业微信群")
    parser.add_argument("report", help="要发送的 Markdown 文件")
    parser.add_argument("--dry-run", action="store_true", help="只检查分段，不实际发送")
    return parser.parse_args()


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def split_markdown(text: str, limit: int = MAX_CHUNK_BYTES) -> list[str]:
    if utf8_len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for line in text.splitlines(keepends=True):
        line_size = utf8_len(line)
        if line_size > limit:
            if current:
                chunks.append("".join(current).rstrip())
                current = []
                current_size = 0
            buffer = ""
            for char in line:
                if utf8_len(buffer + char) > limit:
                    chunks.append(buffer.rstrip())
                    buffer = char
                else:
                    buffer += char
            if buffer:
                current = [buffer]
                current_size = utf8_len(buffer)
            continue
        if current and current_size + line_size > limit:
            chunks.append("".join(current).rstrip())
            current = [line]
            current_size = line_size
        else:
            current.append(line)
            current_size += line_size
    if current:
        chunks.append("".join(current).rstrip())
    return [chunk for chunk in chunks if chunk.strip()]


def send_chunk(webhook_url: str, content: str) -> None:
    body = json.dumps(
        {"msgtype": "markdown", "markdown": {"content": content}},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"企业微信发送失败：{exc}") from exc
    if payload.get("errcode") != 0:
        raise RuntimeError(
            f"企业微信返回错误：{payload.get('errcode')} {payload.get('errmsg')}"
        )


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    report_path = Path(args.report).resolve()
    text = report_path.read_text(encoding="utf-8").strip()
    chunks = split_markdown(text)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "chunks": len(chunks),
                    "chunk_bytes": [utf8_len(chunk) for chunk in chunks],
                },
                ensure_ascii=False,
            )
        )
        return 0

    webhook_url = os.environ.get("WECOM_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print(
            "缺少 WECOM_WEBHOOK_URL。请在 .env 中填写企业微信群机器人 Webhook。",
            file=sys.stderr,
        )
        return 2
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        prefix = f"**RWKV Daily Brief（{index}/{total}）**\n" if total > 1 else ""
        send_chunk(webhook_url, prefix + chunk)
        if index < total:
            time.sleep(1)
    print(f"已发送 {total} 段到企业微信群")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

