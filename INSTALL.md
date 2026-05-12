# 安装指南

## 系统要求

- Python 3.10 或更高版本
- Windows 10/11 或 Linux
- 一个可用的 QQ 号（用于 Lagrange）

---

## 步骤 1: 安装 Python 依赖

```bash
cd claude-qq-bot

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -e .
```

---

## 步骤 2: 配置环境变量

编辑 `.env` 文件，确认以下配置：

```env
# 机器人监听地址
HOST=127.0.0.1
PORT=8081

# OneBot / NapCat 反向 WebSocket
ONEBOT_WS_URL=ws://127.0.0.1:8081/onebot/v11/ws

# 中科大 LLM API 配置
LLM_API_BASE=https://api.llm.ustc.edu.cn/v1
LLM_API_KEY=REPLACE_WITH_REAL_LLM_API_KEY
LLM_MODEL=deepseek-v4-pro
LLM_VISION_MODEL=qwen-chat
# 可选：图片模型与文字模型不在同一接口时单独配置
# LLM_VISION_API_BASE=https://api.llm.ustc.edu.cn/v1
# LLM_VISION_API_KEY=REPLACE_WITH_REAL_VISION_API_KEY

# 会话配置
MAX_CONTEXT_MESSAGES=20
SESSION_TIMEOUT=3600

# 系统提示词
SYSTEM_PROMPT=你是一个 QQ 机器人助手，用简洁友好的语气回复用户。
```

图片消息会优先走 `LLM_VISION_MODEL`；如果图片模型调用失败，机器人会回退到文字模型并提示对方补充描述。

`.env` 的加载策略是 `override=False`：如果你在 PowerShell、systemd、Docker 或其他启动进程中已经设置了同名环境变量，旧 `.env` 不会覆盖它们。

---

## 步骤 3: 下载并配置 NapCat/Lagrange

参考 `DEPLOY.md` 详细指南。

简要步骤：
1. 下载并登录 NapCat 或 Lagrange.Core
2. 扫码登录 QQ
3. 配置反向 WebSocket，连接到：

```text
ws://127.0.0.1:8081/onebot/v11/ws
```

---

## 步骤 4: 启动机器人

### Windows

```bash
# 方式 A: 使用启动脚本
start.bat

# 方式 B: 手动启动
python -u bot.py
```

### Linux

```bash
# 方式 A: 使用启动脚本
chmod +x start.sh
./start.sh

# 方式 B: 手动启动
python -u bot.py
```

`nb-cli` 不再是默认依赖。只有你明确想使用 `nb run` 时，才需要自行安装 `nb-cli` 并维护对应配置。

---

## 测试

启动成功后，在 QQ 中：

1. **私聊测试**: 给机器人 QQ 发送 "你好"
2. **群聊测试**: 在群内发送 "@机器人 你好"

预期回复：
```
你好！有什么我可以帮助你的吗？
```

---

## 命令列表

| 命令 | 说明 |
|------|------|
| @机器人 + 消息 | 与机器人对话 |
| /clear | 清空当前会话历史 |

---

## 故障排查

### 虚拟环境创建失败

```bash
# 升级 pip
python -m pip install --upgrade pip

# 重新创建虚拟环境
python -m venv venv --clear
```

### 依赖安装失败

```bash
# 使用国内镜像
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 端口被占用

```bash
# Windows: 查看占用端口的进程
netstat -ano | findstr :8081

# Linux: 
lsof -i :8081
```

修改 `.env` 中的 `PORT` 为其他端口即可。
