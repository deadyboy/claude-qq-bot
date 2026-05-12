# QQ 机器人 - Claude QQ Bot

基于 nonebot2 + OneBot 协议的 QQ 机器人，支持接入中科大 LLM API（Qwen/DeepSeek 等）。

## 功能特性

- ✅ 群聊/私聊支持
- ✅ @机器人 或回复触发
- ✅ 多轮对话记忆（持久化存储）
- ✅ 长文本自动分段
- ✅ 会话清空命令
- ✅ 模型热切换（/model 命令）
- ✅ 支持 Qwen/DeepSeek 等模型

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 配置 Lagrange/NapCat

下载并配置 QQ 机器人客户端：
- **Lagrange.Core**: https://github.com/LagrangeDev/Lagrange.Core
- **NapCat**: https://napcat.dev

当前 `bot.py` 启动的是 NoneBot 反向 WebSocket 服务，监听：

```text
ws://127.0.0.1:8081/onebot/v11/ws
```

在 NapCat/Lagrange 中启用反向 WebSocket，并连接到上面的地址。示例：
```json
{
  "network": {
    "websocket_client": {
      "url": "ws://127.0.0.1:8081/onebot/v11/ws"
    }
  }
}
```

### 3. 配置环境变量

编辑 `.env` 文件：
```env
HOST=127.0.0.1
PORT=8081
ONEBOT_WS_URL=ws://127.0.0.1:8081/onebot/v11/ws
LLM_API_BASE=https://api.llm.ustc.edu.cn/v1
LLM_API_KEY=REPLACE_WITH_REAL_LLM_API_KEY
LLM_MODEL=deepseek-v4-pro
LLM_VISION_MODEL=qwen-chat
# 可选：图片模型与文字模型不在同一接口时单独配置
# LLM_VISION_API_BASE=https://api.llm.ustc.edu.cn/v1
# LLM_VISION_API_KEY=REPLACE_WITH_REAL_VISION_API_KEY
```

收到图片时会优先调用 `LLM_VISION_MODEL`；如果该模型或接口暂时不支持图片，会回退到文字模型并让对方补充文字描述，避免直接报 API 调用失败。

`.env` 只作为本地默认配置加载，不会覆盖启动进程中已经设置的环境变量。

### 4. 启动机器人

```bash
python -u bot.py
```

`nb run` 不是默认启动方式；只有你额外安装并维护 `nb-cli` 时才需要使用它。

## 命令列表

| 命令 | 说明 |
|------|------|
| @机器人 + 消息 | 与机器人对话 |
| /clear | 清空对话历史 |
| /model | 查看当前模型 |
| /model &lt;模型名&gt; | 切换模型（跨供应商时会提示 key/base 兼容性） |

## 项目结构

```
claude-qq-bot/
├── bot.py              # 入口文件
├── .env                # 环境配置
├── src/plugins/claude/
│   ├── __init__.py     # 插件入口
│   ├── api.py          # LLM API 调用
│   ├── memory_core.py  # 会话记忆、用户事实、任务
│   ├── formatter.py    # 消息格式化
│   ├── dialogue.py     # 对话处理
│   └── commands/       # 命令处理器
└── data/sessions/      # 会话数据存储
```
