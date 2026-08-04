# RWKV Discord Daily Brief

每天自动读取 RWKV Discord 中 Bot 有权限查看的活跃文字频道，生成面向 RWKV 架构作者的中文简报，并发布到本 GitHub 私有仓库。

## 日报结构

1. 总览 Brief
2. RWKV 技术相关讨论
3. Bug 与问题
4. 社区反馈
5. General

重点关注开发者的技术意见、待回答问题、文档与工具需求。每个实质性结论必须附 Discord 原消息链接，方便人工核验。
总览开头固定展示过去 24 小时有更新的频道数量和频道名称，然后再进入正文。

## 自动运行方式

GitHub Actions 每天北京时间 08:00 运行：

1. 使用只读 Discord Bot API 抓取前一天 00:00–24:00 的消息；
2. 自动忽略公告、规则、只读频道和当天没有消息的频道；
3. 使用 OpenAI 或兼容中转站 API 生成简报；
4. 仅提交 `reports/YYYY-MM-DD.md`，不上传原始聊天记录；
5. 创建同日期 GitHub Issue，方便团队成员订阅和讨论。

也可以在 GitHub 的 Actions 页面手动运行，并指定需要补跑的日期。

## 必需的 GitHub Actions Secrets

在仓库 `Settings → Secrets and variables → Actions` 中添加：

- `DISCORD_BOT_TOKEN`：Discord Developer Portal 的 Bot Token；
- `LLM_API_KEY`：OpenAI 或中转站 API Key。

可选：

- `DISCORD_GUILD_ID`：通常留空，程序会自动选择唯一服务器或名称包含 RWKV 的服务器；
- `LLM_BASE_URL`：API 基础地址，例如 `https://api.openai.com/v1`；
- `LLM_MODEL`：中转站提供的模型名称；
- `LLM_API_MODE`：`responses` 或 `chat_completions`。

`LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_API_MODE` 是非敏感配置，建议放在仓库的
`Settings → Secrets and variables → Actions → Variables` 中。为保护密钥传输，
`LLM_BASE_URL` 必须使用 HTTPS。

定时计划已在真实端到端运行和人工核验通过后启用；当前使用 `deepseek-v4-flash`，每天北京时间 08:00 生成前一天的日报。

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
