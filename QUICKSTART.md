# 快速启动指南

## 已完成配置

项目已配置完成，可以直接使用！

### 当前 API 配置

```
API 地址：https://api.llm.ustc.edu.cn/v1
API Key: 请在 `.env` 中配置，不要写入文档
默认模型：deepseek-v4-pro
图片模型：qwen-chat
图片模型接口：默认按图片模型自动选择，可用 `LLM_VISION_API_BASE` 单独覆盖
图片模型失败时：自动回退到文字模型，避免直接报 API 调用失败
```

### 可用模型

| 模型名 | 说明 |
|--------|------|
| qwen-chat | 通义千问对话版 |
| qwen2.5-72b | Qwen 2.5 72B |
| deepseek-v4-flash | DeepSeek V4 快速版 |
| deepseek-v4-pro | DeepSeek V4 专业版 |

---

## 启动步骤

### 1. 启动 Lagrange (QQ 桥接)

Windows: 运行 Lagrange.OneBot.exe 或 NapCat。

当前 `bot.py` 启动的是 NoneBot 反向 WebSocket 服务，监听：

```text
ws://127.0.0.1:8081/onebot/v11/ws
```

在 Lagrange/NapCat 中配置反向 WebSocket 连接到上面的地址。

### 2. 启动机器人

```bash
cd F:\ClaudeSpace2\claude-qq-bot
python -u bot.py
```

或直接运行 `run.bat`。

---

## 测试

在 QQ 中：
1. 私聊机器人 QQ 号，发送 "你好"
2. 或在群聊中发送 "@机器人 你好"

---

## 命令

| 命令 | 效果 |
|------|------|
| @机器人 你好 | 与 AI 对话 |
| /clear | 清空对话历史 |
| /model | 查看当前模型 |
| /model deepseek-v4-pro | 切换模型，并显示文字/图片 base 与 key 状态 |

---

## 文件清单

```
claude-qq-bot/
├── bot.py              # 启动入口 ✅
├── pyproject.toml      # 依赖配置 ✅
├── .env                # 环境配置 ✅
├── start.bat           # Windows 启动脚本 ✅
├── start.sh            # Linux 启动脚本 ✅
├── README.md           # 项目说明 ✅
├── INSTALL.md          # 安装指南 ✅
├── DEPLOY.md           # Lagrange 部署指南 ✅
└── src/plugins/claude/
    ├── __init__.py     # 插件入口 ✅
    ├── api.py          # LLM API 调用 ✅
    ├── memory_core.py  # 会话记忆、用户事实、任务 ✅
    ├── formatter.py    # 消息格式化 ✅
    ├── config.py       # 模型配置 ✅
    ├── dialogue.py     # 对话处理 ✅
    └── commands/       # 命令处理器 ✅
```
