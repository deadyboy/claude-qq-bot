@echo off
echo ========================================
echo   QQ 机器人启动脚本
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] 确认 OneBot 客户端配置...
echo 请确保 NapCat 或 Lagrange 已登录，并配置反向 WebSocket:
echo ws://127.0.0.1:8081/onebot/v11/ws

echo.
echo [2/3] 激活虚拟环境...
if not exist "venv" (
    echo [!] 虚拟环境未创建，正在安装依赖...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -e .
) else (
    call venv\Scripts\activate.bat
)

echo.
echo [3/3] 启动机器人...
echo 监听反向 WebSocket: ws://127.0.0.1:8081/onebot/v11/ws
python -u bot.py

pause
