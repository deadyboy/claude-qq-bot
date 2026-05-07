from nonebot import plugin
from nonebot.plugin import PluginMetadata

__plugin_meta__ = PluginMetadata(
    name="claude",
    description="QQ 机器人 - 基于中科大 LLM API 的对话系统",
    usage="与 AI 对话，支持群聊和私聊",
)

# 导入对话处理模块（注册 on_message 事件）
from . import dialogue
