#!/bin/bash

echo "========================================"
echo "  QQ 机器人启动脚本"
echo "========================================"
echo

cd "$(dirname "$0")"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "[!] 虚拟环境未创建，正在安装依赖..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
else
    echo "[OK] 激活虚拟环境..."
    source venv/bin/activate
fi

echo "[OK] 启动机器人..."
echo "监听反向 WebSocket: ws://127.0.0.1:8081/onebot/v11/ws"
python -u bot.py
