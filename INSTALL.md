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
PORT=8080

# OneBot WebSocket 地址（Lagrange 默认端口 6700）
ONEBOT_WS_URL=ws://127.0.0.1:6700

# 中科大 LLM API 配置
LLM_API_BASE=https://api.llm.ustc.edu.cn/v1
LLM_API_KEY=sk-32FTdpTEiL24atPrZxO9yg
LLM_MODEL=qwen-chat

# 会话配置
MAX_CONTEXT_MESSAGES=20
SESSION_TIMEOUT=3600

# 系统提示词
SYSTEM_PROMPT=你是一个 QQ 机器人助手，用简洁友好的语气回复用户。
```

---

## 步骤 3: 下载并配置 Lagrange

参考 `DEPLOY.md` 详细指南。

简要步骤：
1. 下载 Lagrange.Core
2. 扫码登录 QQ
3. 配置 WebSocket 端口 6700

---

## 步骤 4: 启动机器人

### Windows

```bash
# 方式 A: 使用启动脚本
start.bat

# 方式 B: 手动启动
nb run
```

### Linux

```bash
# 方式 A: 使用启动脚本
chmod +x start.sh
./start.sh

# 方式 B: 手动启动
nb run
```

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
netstat -ano | findstr :8080

# Linux: 
lsof -i :8080
```

修改 `.env` 中的 `PORT` 为其他端口即可。
