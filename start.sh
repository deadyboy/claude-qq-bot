#!/bin/bash

echo "========================================"
echo "  QQ 机器人启动脚本"
echo "========================================"
echo

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
nb run
