# RWKV Discord Daily Brief

每天自动读取 RWKV Discord 中 Bot 有权限查看的活跃文字频道，生成面向 RWKV 架构团队的中文简报，并发布到本 GitHub 私有仓库。

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
3. 使用 DeepSeek API 生成简报；
4. 仅提交 `reports/YYYY-MM-DD.md`，不上传原始聊天记录；
5. 创建同日期 GitHub Issue，方便团队成员订阅和讨论。

也可以在 GitHub 的 Actions 页面手动运行，并指定需要补跑的日期。
