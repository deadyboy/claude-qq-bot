# Lagrange 部署指南

## 步骤 1: 下载 Lagrange.Core

### Windows 用户

1. 访问 releases 页面：https://github.com/LagrangeDev/Lagrange.Core/releases
2. 下载最新版本的 `Lagrange.Core.Windows.zip`
3. 解压到任意目录，例如 `D:\Lagrange`

### Linux 用户

```bash
git clone https://github.com/LagrangeDev/Lagrange.Core.git
cd Lagrange.Core
dotnet build -c Release
```

---

## 步骤 2: 配置 Lagrange

### 方式 A: 使用 WebUI (推荐)

1. 运行 `Lagrange.OneBot.exe` (Windows) 或 `dotnet Lagrange.OneBot.dll` (Linux)
2. 首次运行会提示扫码登录
3. 登录成功后，在 `appsettings.json` 中配置：

```json
{
  "OneBot": {
    "Host": "127.0.0.1",
    "Port": 6700,
    "AccessToken": "",
    "UseUniversal": true,
    "ReconnectInterval": 5,
    "MessagePostFormat": "array"
  }
}
```

### 方式 B: 配置文件

编辑 `appsettings.json`:

```json
{
  "OneBot": {
    "Host": "127.0.0.1",
    "Port": 6700,
    "AccessToken": "",
    "UseReverseWebSocket": false,
    "MessagePostFormat": "array"
  },
  "SignServer": {
    "SignServers": []
  }
}
```

---

## 步骤 3: 配置 WebSocket 连接

Lagrange 默认作为 **WebSocket 服务端**，nonebot 作为客户端连接。

在 `appsettings.json` 中确认：

```json
{
  "OneBot": {
    "Host": "127.0.0.1",
    "Port": 6700
  }
}
```

然后在 nonebot 配置中连接：

```python
# 在 bot.py 或 .env 中
ONEBOT_WS_URL=ws://127.0.0.1:6700
```

---

## 步骤 4: 启动顺序

1. **先启动 Lagrange**
   ```bash
   # Windows
   Lagrange.OneBot.exe
   
   # Linux
   dotnet Lagrange.OneBot.dll
   ```

2. **再启动 nonebot 机器人**
   ```bash
   cd claude-qq-bot
   nb run
   ```

---

## 验证连接

启动成功后，你应该看到：

```
[INFO] nonebot: OneBot V11 connected
[INFO] nonebot: Bot self_id=12345678 connected
```

在 QQ 中发送消息测试：
- 私聊机器人 QQ 号
- 群聊中 `@机器人 你好`

---

## 常见问题

### Q: 扫码失败/登录失败
A: 可能是账号风控，尝试：
- 使用常用设备登录
- 先在手机 QQ 登录同一账号
- 等待 24 小时后再试

### Q: 消息收不到
A: 检查：
- Lagrange 和 nonebot 是否都启动
- WebSocket 端口 6700 是否开放
- 防火墙是否拦截

### Q: 群聊不回复
A: 确认：
- 群消息是否被 @
- 机器人是否在群内
- 消息过滤器设置

---

## 替代方案：NapCat

如果 Lagrange 不稳定，可尝试 NapCat:
- 官网：https://napcat.dev
- 支持 Windows/Mac/Linux
- 配置更简单，图形界面友好
