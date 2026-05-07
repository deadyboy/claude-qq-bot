@echo off
echo ========================================
echo   QQ 机器人启动脚本
echo ========================================
echo.

echo [1/3] 检查 Lagrange 是否运行...
tasklist /FI "WINDOWTITLE eq Lagrange*" /NH 2>nul | find /I "Lagrange" >nul
if errorlevel 1 (
    echo [!] Lagrange 未运行，请先启动 Lagrange.OneBot.exe
    echo     路径：D:\Lagrange\Lagrange.OneBot.exe
    pause
    exit /b 1
)
echo [OK] Lagrange 运行中

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
nb run

pause
