# RWKV Discord Daily Brief

每天自动读取 RWKV Discord 中 Bot 有权限查看的活跃文字频道，生成面向 RWKV 架构作者的中文简报，并发布到本 GitHub 私有仓库。

## 日报结构

1. 总览 Brief
2. RWKV 技术相关讨论
3. Bug 与问题
4. 社区反馈
5. General

重点关注开发者的技术意见、待回答问题、文档与工具需求。每个实质性结论必须附 Discord 原消息链接，方便人工核验。

## 自动运行方式

GitHub Actions 每天北京时间 08:00 运行：

1. 使用只读 Discord Bot API 抓取前一天 00:00–24:00 的消息；
2. 自动忽略公告、规则、只读频道和当天没有消息的频道；
3. 使用 OpenAI Responses API 生成简报；
4. 仅提交 `reports/YYYY-MM-DD.md`，不上传原始聊天记录；
5. 创建同日期 GitHub Issue，方便团队成员订阅和讨论。

也可以在 GitHub 的 Actions 页面手动运行，并指定需要补跑的日期。

## 必需的 GitHub Actions Secrets

在仓库 `Settings → Secrets and variables → Actions` 中添加：

- `DISCORD_BOT_TOKEN`：Discord Developer Portal 的 Bot Token；
- `OPENAI_API_KEY`：OpenAI Platform API Key。

可选：

- `DISCORD_GUILD_ID`：通常留空，程序会自动选择唯一服务器或名称包含 RWKV 的服务器；
- `OPENAI_MODEL`：默认 `gpt-5.6-luna`，适合低成本日报总结。

不要把真实密钥写进代码、Issue、聊天或报告。

## 频道策略

配置文件：`config/channel_rules.json`

- 默认读取 Bot 可见的文字频道与帖子；
- 默认跳过公告频道、规则、欢迎、只读等频道；
- 当天没有消息的频道不会进入模型上下文；
- 可用 `exclude_channel_ids` 排除指定频道；
- 可用 `include_channel_ids` 只监控指定频道。

## 本地验证

```powershell
python -m unittest discover -s tests -v
python src/discord_daily.py --date 2026-08-02
python src/generate_daily_report.py work/discord_messages_2026-08-02.json
```

本地 `.env` 已被 `.gitignore` 排除，不会上传。
