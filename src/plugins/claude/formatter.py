"""消息格式转换模块"""

import re
import os
from typing import List

# QQ 消息长度上限
QQ_MSG_LIMIT = 2000


def split_qq_msg(text: str) -> List[str]:
    """将长文本按 QQ 消息限制拆分"""
    if len(text) <= QQ_MSG_LIMIT:
        return [text]

    messages = []
    while text:
        if len(text) <= QQ_MSG_LIMIT:
            messages.append(text)
            break
        # 按行拆分
        lines = text.split("\n")
        current = []
        for line in lines:
            if len("\n".join(current + [line])) <= QQ_MSG_LIMIT:
                current.append(line)
            else:
                if current:
                    messages.append("\n".join(current))
                current = [line]
        text = "\n".join(current)
        if text and len(text) > QQ_MSG_LIMIT:
            messages.append(text[:QQ_MSG_LIMIT])
            text = text[QQ_MSG_LIMIT:]

    return messages or [text]


def contains_cq_code(text: str) -> bool:
    """检查是否包含 CQ 码"""
    return bool(re.search(r"\[CQ:.*?\]", text))


def extract_cq_image_urls(text: str) -> List[str]:
    """从消息中提取图片 URL"""
    pattern = r"\[CQ:image,file=([^,\]]+),.*?\]"
    return re.findall(pattern, text)


def format_reply(content: str) -> str:
    """清理 Claude 回复中的系统标记"""
    # 移除可能存在的 XML 标记
    content = re.sub(r"<\|.*?\|>", "", content)
    # 移除多余空白
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def sanitize_for_qq_text(content: str) -> str:
    """移除部分 QQ/Windows 日志链路容易出问题的特殊字符。"""
    # Strip emoji and other non-BMP characters, plus common BMP emoji blocks.
    content = "".join(ch for ch in content if ord(ch) <= 0xFFFF)
    content = re.sub(
        r"[\u200d\ufe0e\ufe0f\u2600-\u27bf\u2b00-\u2bff]",
        "",
        content,
    )
    return content.strip()
