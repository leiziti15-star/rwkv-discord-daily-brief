#!/usr/bin/env python3
"""Collect a day's Discord messages using a read-only bot token.

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


DISCORD_API = "https://discord.com/api/v10"
DISCORD_EPOCH_MS = 1420070400000
TEXT_CHANNEL = 0
ANNOUNCEMENT_CHANNEL = 5
PUBLIC_THREAD = 11
PRIVATE_THREAD = 12
ANNOUNCEMENT_THREAD = 10
FORUM_CHANNEL = 15
MEDIA_CHANNEL = 16
CATEGORY_CHANNEL = 4
THREAD_TYPES = {ANNOUNCEMENT_THREAD, PUBLIC_THREAD, PRIVATE_THREAD}
PARENT_TYPES = {TEXT_CHANNEL, FORUM_CHANNEL, MEDIA_CHANNEL}


class DiscordAPIError(RuntimeError):
    pass


@dataclass
class Window:
    report_date: date
    start_local: datetime
    end_local: datetime
    start_utc: datetime
    end_utc: datetime


class DiscordClient:
    def __init__(self, token: str) -> None:
        self.token = token

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{DISCORD_API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        for attempt in range(6):
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bot {self.token}",
                    "User-Agent": "RWKVDailyBrief/1.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < 5:
                    try:
                        retry_after = float(json.loads(body).get("retry_after", 1))
                    except (ValueError, json.JSONDecodeError):
                        retry_after = 1
                    time.sleep(min(max(retry_after, 0.25), 30))
                    continue
                if exc.code in {403, 404}:
                    raise DiscordAPIError(f"Discord 拒绝访问 {path}（HTTP {exc.code}）") from exc
                raise DiscordAPIError(
                    f"Discord API 请求失败：HTTP {exc.code}，{body[:300]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 5:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise DiscordAPIError(f"无法连接 Discord：{exc.reason}") from exc
        raise DiscordAPIError("Discord API 多次重试后仍未成功")


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding existing environment values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 RWKV Discord 每日聊天记录")
    parser.add_argument(
        "--date",
        help="报告日期，格式 YYYY-MM-DD；默认是 Asia/Shanghai 的昨天",
    )
    parser.add_argument(
        "--output-dir",
        default="work",
        help="原始记录输出目录，默认 work",
    )
    return parser.parse_args()


def report_window(date_text: str | None, timezone_name: str) -> Window:
    tz = ZoneInfo(timezone_name)
    if date_text:
        report_date = date.fromisoformat(date_text)
    else:
        report_date = datetime.now(tz).date() - timedelta(days=1)
    start_local = datetime.combine(report_date, dt_time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return Window(
        report_date=report_date,
        start_local=start_local,
        end_local=end_local,
        start_utc=start_local.astimezone(timezone.utc),
        end_utc=end_local.astimezone(timezone.utc),
    )


def datetime_to_snowflake(value: datetime) -> int:
    milliseconds = int(value.timestamp() * 1000)
    return max(0, milliseconds - DISCORD_EPOCH_MS) << 22


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def read_rules(path: Path) -> dict[str, Any]:
    default = {
        "include_channel_ids": [],
        "exclude_channel_ids": [],
        "skip_channel_types": [ANNOUNCEMENT_CHANNEL],
        "skip_name_patterns": [],
        "include_bot_messages": False,
        "max_messages_per_channel": 1000,
        "scan_archived_threads": True,
    }
    if path.exists():
        default.update(json.loads(path.read_text(encoding="utf-8")))
    return default


def choose_guild(
    guilds: list[dict[str, Any]], guild_id: str, name_hint: str
) -> dict[str, Any]:
    if guild_id:
        for guild in guilds:
            if str(guild.get("id")) == guild_id:
                return guild
        raise DiscordAPIError("Bot 当前不在 DISCORD_GUILD_ID 指定的服务器中")
    if len(guilds) == 1:
        return guilds[0]
    if name_hint:
        lowered = name_hint.casefold()
        matches = [
            guild
            for guild in guilds
            if lowered in str(guild.get("name", "")).casefold()
        ]
        if len(matches) == 1:
            return matches[0]
    names = "、".join(str(guild.get("name", guild.get("id"))) for guild in guilds)
    raise DiscordAPIError(
        "无法自动确定 RWKV 服务器。请填写 DISCORD_GUILD_ID。"
        + (f" Bot 当前可见：{names}" if names else " Bot 当前没有加入任何服务器。")
    )


def channel_reason(channel: dict[str, Any], rules: dict[str, Any]) -> tuple[bool, str]:
    channel_id = str(channel["id"])
    include_ids = {str(item) for item in rules.get("include_channel_ids", [])}
    exclude_ids = {str(item) for item in rules.get("exclude_channel_ids", [])}
    if channel_id in exclude_ids:
        return False, "手动排除"
    if include_ids:
        return (channel_id in include_ids, "手动纳入" if channel_id in include_ids else "不在手动纳入列表")
    if int(channel.get("type", -1)) in set(rules.get("skip_channel_types", [])):
        return False, "只读/公告频道类型"
    name = channel.get("name", "")
    for pattern in rules.get("skip_name_patterns", []):
        if re.search(pattern, name, flags=re.IGNORECASE):
            return False, f"名称匹配排除规则：{pattern}"
    if int(channel.get("type", -1)) not in PARENT_TYPES | THREAD_TYPES:
        return False, "非文字频道"
    return True, "可采集"


def get_archived_threads(
    client: DiscordClient,
    parent: dict[str, Any],
    start_snowflake: int,
) -> list[dict[str, Any]]:
    try:
        payload = client.get(
            f"/channels/{parent['id']}/threads/archived/public",
            {"limit": 100},
        )
    except DiscordAPIError:
        return []
    result = []
    for thread in payload.get("threads", []):
        last_message_id = thread.get("last_message_id")
        if last_message_id and int(last_message_id) >= start_snowflake:
            result.append(thread)
    return result


def collect_threads(
    client: DiscordClient,
    guild_id: str,
    parents: list[dict[str, Any]],
    rules: dict[str, Any],
    start_snowflake: int,
) -> list[dict[str, Any]]:
    threads: dict[str, dict[str, Any]] = {}
    try:
        active_payload = client.get(f"/guilds/{guild_id}/threads/active")
        for thread in active_payload.get("threads", []):
            threads[str(thread["id"])] = thread
    except DiscordAPIError:
        pass

    if rules.get("scan_archived_threads", True):
        for parent in parents:
            for thread in get_archived_threads(client, parent, start_snowflake):
                threads[str(thread["id"])] = thread
    return list(threads.values())


def fetch_messages(
    client: DiscordClient,
    channel_id: str,
    window: Window,
    maximum: int,
) -> list[dict[str, Any]]:
    start_snowflake = datetime_to_snowflake(window.start_utc)
    cursor_before = datetime_to_snowflake(window.end_utc)
    collected: list[dict[str, Any]] = []

    while len(collected) < maximum:
        payload = client.get(
            f"/channels/{channel_id}/messages",
            {"limit": min(100, maximum - len(collected)), "before": str(cursor_before)},
        )
        if not payload:
            break
        page_ids = [int(item["id"]) for item in payload]
        for message in payload:
            timestamp = parse_timestamp(message["timestamp"])
            if window.start_utc <= timestamp < window.end_utc:
                collected.append(message)
        oldest_id = min(page_ids)
        if oldest_id <= start_snowflake or len(payload) < 100:
            break
        cursor_before = oldest_id

    collected.sort(key=lambda item: int(item["id"]))
    return collected


def safe_text(value: str, limit: int = 4000) -> str:
    value = value.replace("\x00", "").strip()
    return value[:limit]


def normalise_message(
    message: dict[str, Any],
    guild_id: str,
    channel: dict[str, Any],
    channel_lookup: dict[str, dict[str, Any]],
    timezone_name: str,
) -> dict[str, Any]:
    tz = ZoneInfo(timezone_name)
    author = message.get("author", {})
    parent = channel_lookup.get(str(channel.get("parent_id", "")))
    category = None
    parent_name = None
    if parent:
        if int(parent.get("type", -1)) == CATEGORY_CHANNEL:
            category = parent.get("name")
        else:
            parent_name = parent.get("name")
            grandparent = channel_lookup.get(str(parent.get("parent_id", "")))
            if grandparent and int(grandparent.get("type", -1)) == CATEGORY_CHANNEL:
                category = grandparent.get("name")

    content = safe_text(message.get("content", ""))
    attachments = [
        {
            "filename": item.get("filename"),
            "url": item.get("url"),
            "content_type": item.get("content_type"),
        }
        for item in message.get("attachments", [])
    ]
    embeds = [
        {
            "title": safe_text(item.get("title", ""), 500),
            "description": safe_text(item.get("description", ""), 1000),
            "url": item.get("url"),
        }
        for item in message.get("embeds", [])[:5]
    ]
    reactions = [
        {"emoji": item.get("emoji", {}).get("name"), "count": item.get("count", 0)}
        for item in message.get("reactions", [])
    ]
    timestamp_utc = parse_timestamp(message["timestamp"])
    return {
        "id": str(message["id"]),
        "channel_id": str(channel["id"]),
        "channel_name": channel.get("name", str(channel["id"])),
        "channel_type": int(channel.get("type", -1)),
        "parent_name": parent_name,
        "category_name": category,
        "timestamp": timestamp_utc.isoformat(),
        "timestamp_local": timestamp_utc.astimezone(tz).isoformat(),
        "author": {
            "id": str(author.get("id", "")),
            "username": author.get("global_name") or author.get("username") or "unknown",
            "bot": bool(author.get("bot", False)),
        },
        "content": content,
        "attachments": attachments,
        "embeds": embeds,
        "reactions": reactions,
        "reply_to_message_id": (
            message.get("message_reference", {}).get("message_id")
        ),
        "url": (
            f"https://discord.com/channels/{guild_id}/{channel['id']}/{message['id']}"
        ),
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Discord 原始记录 | {payload['report_date']}",
        "",
        f"- 时间范围：{payload['window']['start_local']} 至 {payload['window']['end_local']}",
        f"- 消息数：{payload['stats']['message_count']}",
        f"- 活跃频道数：{payload['stats']['active_channel_count']}",
        "",
    ]
    messages = payload["messages"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for message in messages:
        group = message.get("category_name") or "未分组"
        channel = message.get("channel_name") or message["channel_id"]
        grouped.setdefault((group, channel), []).append(message)
    for (group, channel), items in sorted(grouped.items()):
        lines.extend([f"## {group} / #{channel}", ""])
        for item in items:
            local_time = item["timestamp_local"][11:16]
            content = item["content"].replace("\n", " ")
            if not content and item["attachments"]:
                content = "附件：" + ", ".join(
                    attachment.get("filename") or "未命名附件"
                    for attachment in item["attachments"]
                )
            if not content and item["embeds"]:
                content = "嵌入内容：" + "；".join(
                    embed.get("title") or embed.get("description") or "链接"
                    for embed in item["embeds"]
                )
            lines.append(
                f"- {local_time} **{item['author']['username']}**："
                f"{content[:1000]} ([原消息]({item['url']}))"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    guild_id = os.environ.get("DISCORD_GUILD_ID", "").strip()
    guild_name = os.environ.get("DISCORD_GUILD_NAME", "RWKV").strip()
    timezone_name = os.environ.get("TIMEZONE", "Asia/Shanghai")
    rules_path = root / os.environ.get(
        "CHANNEL_RULES_PATH", "config/channel_rules.json"
    )
    if not token:
        print(
            "缺少 DISCORD_BOT_TOKEN。请在 .env 中填写 Bot Token。",
            file=sys.stderr,
        )
        return 2

    window = report_window(args.date, timezone_name)
    rules = read_rules(rules_path)
    client = DiscordClient(token)
    guilds = client.get("/users/@me/guilds")
    try:
        selected_guild = choose_guild(guilds, guild_id, guild_name)
    except DiscordAPIError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    guild_id = str(selected_guild["id"])
    channels = client.get(f"/guilds/{guild_id}/channels")
    channel_lookup = {str(channel["id"]): channel for channel in channels}

    inventory = []
    parents = []
    for channel in channels:
        included, reason = channel_reason(channel, rules)
        inventory.append(
            {
                "id": str(channel["id"]),
                "name": channel.get("name"),
                "type": int(channel.get("type", -1)),
                "parent_id": channel.get("parent_id"),
                "included": included,
                "reason": reason,
                "last_message_id": channel.get("last_message_id"),
            }
        )
        if included and int(channel.get("type", -1)) in PARENT_TYPES:
            parents.append(channel)

    start_snowflake = datetime_to_snowflake(window.start_utc)
    threads = collect_threads(client, guild_id, parents, rules, start_snowflake)
    for thread in threads:
        channel_lookup[str(thread["id"])] = thread

    candidates = list(parents)
    for thread in threads:
        parent = channel_lookup.get(str(thread.get("parent_id", "")))
        if not parent:
            continue
        parent_allowed, _ = channel_reason(parent, rules)
        if parent_allowed:
            candidates.append(thread)

    messages = []
    warnings = []
    include_bots = bool(rules.get("include_bot_messages", False))
    maximum = int(rules.get("max_messages_per_channel", 1000))
    for channel in candidates:
        last_message_id = channel.get("last_message_id")
        if last_message_id and int(last_message_id) < start_snowflake:
            continue
        try:
            raw_messages = fetch_messages(
                client, str(channel["id"]), window, maximum
            )
        except DiscordAPIError as exc:
            warnings.append(
                {"channel_id": str(channel["id"]), "warning": str(exc)}
            )
            continue
        for message in raw_messages:
            if not include_bots and message.get("author", {}).get("bot"):
                continue
            normalised = normalise_message(
                message, guild_id, channel, channel_lookup, timezone_name
            )
            if (
                normalised["content"]
                or normalised["attachments"]
                or normalised["embeds"]
            ):
                messages.append(normalised)

    messages.sort(key=lambda item: item["timestamp"])
    active_channel_ids = {item["channel_id"] for item in messages}
    active_users = {item["author"]["id"] for item in messages if item["author"]["id"]}
    payload = {
        "schema_version": 1,
        "guild_id": guild_id,
        "guild_name": selected_guild.get("name"),
        "report_date": window.report_date.isoformat(),
        "timezone": timezone_name,
        "window": {
            "start_local": window.start_local.isoformat(),
            "end_local": window.end_local.isoformat(),
            "start_utc": window.start_utc.isoformat(),
            "end_utc": window.end_utc.isoformat(),
        },
        "stats": {
            "message_count": len(messages),
            "active_channel_count": len(active_channel_ids),
            "active_user_count": len(active_users),
        },
        "messages": messages,
        "channel_inventory": inventory,
        "warnings": warnings,
    }

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"discord_messages_{window.report_date.isoformat()}.json"
    md_path = output_dir / f"discord_messages_{window.report_date.isoformat()}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(payload, md_path)
    print(
        json.dumps(
            {
                "report_date": payload["report_date"],
                "message_count": len(messages),
                "active_channel_count": len(active_channel_ids),
                "json": str(json_path),
                "markdown": str(md_path),
                "warnings": len(warnings),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
