"""自动用户画像抽取。

该模块只负责从用户自然语言中提取稳定、低风险的个人资料。
显式的“记住/忘记”命令仍由 dialogue.py 处理。
"""

import json
import re
from typing import Any, Dict, List

from .api import llm_client


MAX_AUTO_MEMORY_TEXT_LENGTH = 500
MIN_AUTO_MEMORY_CONFIDENCE = 0.68

SENSITIVE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(api[-_ ]?key|access[-_ ]?token|secret|private[-_ ]?key)\b"),
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(密码|口令|密钥|令牌|验证码|身份证|银行卡|私钥|助记词)"),
)

AUTO_MEMORY_HINTS = (
    "我叫",
    "我的名字",
    "以后叫我",
    "以后称呼我",
    "我喜欢",
    "我不喜欢",
    "我的偏好",
    "我习惯",
    "我常用",
    "我希望你",
    "我是一名",
    "我是一个",
    "我是一位",
    "我的职业",
    "我从事",
    "我住在",
    "我来自",
    "我的学校",
    "我的研究方向",
)

UNSTABLE_OBJECT_PREFIXES = (
    "因为",
    "为了",
    "来",
    "想",
    "问",
    "觉得",
    "以为",
    "正在",
    "刚才",
)

EXTRACTION_SYSTEM_PROMPT = """你是一个保守的用户记忆抽取器。
只从“当前用户本人”的消息中提取长期稳定、对以后回复有帮助的信息。

可以提取：
- 称呼、名字、职业、学校、所在地、研究方向
- 长期偏好、习惯、希望助手采用的回复风格
- 与项目协作相关的稳定背景

不要提取：
- 密码、token、API key、验证码、身份证、银行卡等敏感信息
- 一次性请求、临时情绪、闲聊寒暄、问题本身
- 关于其他人的资料，除非用户明确说这是自己资料
- 不确定或需要推断的信息

只返回 JSON 数组，不要 Markdown，不要解释。
数组元素格式：
{"predicate":"2到10字的中文键名","object":"80字以内的值","confidence":0.0}
如果没有值得记住的信息，返回 []。
"""


def contains_sensitive_content(text: str) -> bool:
    """判断文本是否包含不应自动保存的敏感内容。"""
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def should_attempt_auto_memory(text: str) -> bool:
    """判断是否值得调用自动抽取。"""
    stripped = text.strip()
    if not stripped or len(stripped) < 4:
        return False
    if len(stripped) > MAX_AUTO_MEMORY_TEXT_LENGTH:
        return False
    if stripped.startswith("/"):
        return False
    if stripped.startswith(("记住", "忘记")):
        return False
    if contains_sensitive_content(stripped):
        return False
    return any(hint in stripped for hint in AUTO_MEMORY_HINTS)


def _clean_value(value: str) -> str:
    value = value.strip().strip("：:，,。.!！?？ ")
    return re.split(r"[，,。.!！?？\n]", value, maxsplit=1)[0].strip()


def heuristic_extract_facts(text: str) -> List[Dict[str, Any]]:
    """用少量明确模式做快速抽取，降低对 LLM 的依赖。"""
    stripped = text.strip()
    patterns = (
        (r"(?:以后)?(?:请)?(?:叫我|称呼我)(?P<value>[^，,。.!！?？\n]{1,30})", "称呼", ""),
        (r"我叫(?P<value>[^，,。.!！?？\n]{1,30})", "称呼", ""),
        (r"我的名字(?:是|叫)(?P<value>[^，,。.!！?？\n]{1,30})", "称呼", ""),
        (r"我(?:喜欢|偏好)(?P<value>[^，,。.!！?？\n]{1,60})", "偏好", "喜欢"),
        (r"我不喜欢(?P<value>[^，,。.!！?？\n]{1,60})", "偏好", "不喜欢"),
        (r"我(?:是一名|是一个|是一位)(?P<value>[^，,。.!！?？\n]{1,60})", "身份", ""),
        (r"我的职业是(?P<value>[^，,。.!！?？\n]{1,60})", "职业", ""),
        (r"我从事(?P<value>[^，,。.!！?？\n]{1,60})", "职业", ""),
        (r"我住在(?P<value>[^，,。.!！?？\n]{1,40})", "所在地", ""),
        (r"我来自(?P<value>[^，,。.!！?？\n]{1,40})", "所在地", ""),
    )

    facts: List[Dict[str, Any]] = []
    for pattern, predicate, prefix in patterns:
        for match in re.finditer(pattern, stripped):
            value = _clean_value(match.group("value"))
            if value:
                facts.append({
                    "predicate": predicate,
                    "object": f"{prefix}{value}" if prefix else value,
                    "confidence": 0.9,
                })

    return normalize_extracted_facts(facts)


def extract_json_payload(response: str) -> Any:
    """从模型回复中提取 JSON。支持纯 JSON 或 fenced code block。"""
    text = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        array_match = re.search(r"\[.*\]", text, flags=re.S)
        if not array_match:
            raise
        return json.loads(array_match.group(0))


def normalize_extracted_facts(raw_facts: Any) -> List[Dict[str, Any]]:
    """清洗模型输出，避免保存低置信度、敏感或畸形事实。"""
    if isinstance(raw_facts, dict):
        raw_facts = raw_facts.get("facts", [])
    if not isinstance(raw_facts, list):
        return []

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_facts:
        if not isinstance(item, dict):
            continue

        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not predicate or not obj:
            continue
        if confidence < MIN_AUTO_MEMORY_CONFIDENCE:
            continue
        if len(predicate) > 20 or len(obj) > 80:
            continue
        if contains_sensitive_content(f"{predicate} {obj}"):
            continue
        if obj.startswith(UNSTABLE_OBJECT_PREFIXES):
            continue

        key = (predicate, obj)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
        })

    return normalized[:5]


async def extract_user_facts(text: str) -> List[Dict[str, Any]]:
    """抽取用户事实。先用规则兜底，再让 LLM 做更细的判断。"""
    if not should_attempt_auto_memory(text):
        return []

    heuristic_facts = heuristic_extract_facts(text)

    try:
        response = await llm_client.chat(
            messages=[{"role": "user", "content": f"用户消息：{text}"}],
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=0.1,
        )
        llm_facts = normalize_extracted_facts(extract_json_payload(response))
    except Exception:
        return heuristic_facts

    merged = heuristic_facts + llm_facts
    return normalize_extracted_facts(merged)
