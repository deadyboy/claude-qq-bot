@echo off
chcp 65001 >nul
echo ========================================
echo   QQ 机器人启动脚本
echo ========================================
echo.

echo [1/2] 检查 Python 环境...
where python >nul 2>&1
if errorlevel 1 (
    echo [!] Python 未找到，请确保已安装 Python 并添加到 PATH
    pause
    exit /b 1
)
echo [OK] Python 已找到

echo.
echo [2/2] 启动机器人...
echo 监听反向 WebSocket: ws://127.0.0.1:8081/onebot/v11/ws
echo 按 Ctrl+C 停止机器人
echo.

cd /d "%~dp0"
python -u bot.py

pause
