import sys


def _configure_stdio():
    """Keep Windows console logging from crashing on emoji/non-GBK text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


_configure_stdio()

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotAdapter

# 初始化 nonebot
nonebot.init(
    driver="~fastapi",
    host="127.0.0.1",
    port=8081,
)

# 注册 OneBot 适配器
driver = nonebot.get_driver()
driver.register_adapter(OneBotAdapter)

# 加载插件
nonebot.load_plugins("src/plugins")

if __name__ == "__main__":
    nonebot.run()
