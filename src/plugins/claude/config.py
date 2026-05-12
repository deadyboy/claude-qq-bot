"""模型切换配置模块 - 支持运行时切换模型"""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_env() -> None:
    """Load project .env after process env, so shell/service values win."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


load_project_env()

# 支持的模型列表
SUPPORTED_MODELS = {
    "qwen-chat": "https://api.llm.ustc.edu.cn/v1",
    "qwen2.5-72b": "https://api.llm.ustc.edu.cn/v1",
    "deepseek-v4-flash": "https://api.llm.ustc.edu.cn/v1",
    "deepseek-v4-pro": "https://api.llm.ustc.edu.cn/v1",
    "deepseek-chat": "https://api.deepseek.com/v1",
}

PLACEHOLDER_KEY_MARKERS = (
    "replace",
    "placeholder",
    "your_api_key",
    "your-api-key",
    "api_key_here",
    "example",
    "xxxxx",
)


def is_placeholder_api_key(value: Optional[str]) -> bool:
    """Return whether an API key is missing or still looks like sample text."""
    key = (value or "").strip()
    if not key:
        return True

    lowered = key.lower()
    compact = lowered.replace("-", "_").replace(" ", "_")
    if compact in {"key", "api_key", "llm_api_key", "none", "null"}:
        return True
    return any(marker in compact for marker in PLACEHOLDER_KEY_MARKERS)


def validate_api_key_value(value: Optional[str], env_name: str = "LLM_API_KEY") -> str:
    """Return a valid key or raise a clear configuration error."""
    key = (value or "").strip()
    if not key:
        raise RuntimeError(
            f"{env_name} 未配置。请在进程环境或项目 .env 中设置真实 API Key；"
            "本项目加载 .env 时不会覆盖已经存在的进程环境变量。"
        )
    if is_placeholder_api_key(key):
        raise RuntimeError(
            f"{env_name} 仍是占位值。请替换为真实 API Key，文档示例不要使用 sk-... 形态。"
        )
    return key


def format_api_key_state(value: Optional[str]) -> str:
    if not (value or "").strip():
        return "未配置"
    if is_placeholder_api_key(value):
        return "占位值"
    return "已配置"


def provider_from_base_url(base_url: Optional[str]) -> str:
    parsed = urlparse(base_url or "")
    host = parsed.netloc.lower()
    if not host:
        return "未配置"
    if "llm.ustc.edu.cn" in host:
        return "中科大 LLM"
    if "deepseek.com" in host:
        return "DeepSeek"
    return f"自定义({host})"


class ModelConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = os.getenv("LLM_MODEL", "deepseek-v4-pro")
        self.vision_model = os.getenv("LLM_VISION_MODEL", "qwen-chat")
        self._api_base_from_env = bool(os.getenv("LLM_API_BASE"))
        self._vision_api_base_from_env = bool(os.getenv("LLM_VISION_API_BASE"))
        self._vision_api_key_from_env = bool(os.getenv("LLM_VISION_API_KEY"))
        self.api_base = os.getenv("LLM_API_BASE") or SUPPORTED_MODELS.get(
            self.model,
            "https://api.llm.ustc.edu.cn/v1"
        )
        self.vision_api_base = os.getenv("LLM_VISION_API_BASE") or SUPPORTED_MODELS.get(
            self.vision_model,
            self.api_base,
        )
        self.api_key = os.getenv("LLM_API_KEY")
        self.vision_api_key = os.getenv("LLM_VISION_API_KEY") or self.api_key
        self._initialized = True

    def switch_model(self, model_name: str) -> tuple[bool, str]:
        """切换模型"""
        if model_name not in SUPPORTED_MODELS:
            available = ", ".join(SUPPORTED_MODELS.keys())
            return False, f"不支持的模型：{model_name}。可用：{available}"

        old_provider = provider_from_base_url(self.api_base)
        self.model = model_name
        self.api_base = SUPPORTED_MODELS[model_name]
        new_provider = provider_from_base_url(self.api_base)
        os.environ["LLM_MODEL"] = model_name
        os.environ["LLM_API_BASE"] = self.api_base

        lines = [
            f"已切换到模型：{model_name}",
            f"API Base：{self.api_base}（{new_provider}）",
            f"文字 Key：{self.get_api_key_state()}",
            f"图片模型：{self.vision_model}",
            f"图片 API Base：{self.vision_api_base}（{provider_from_base_url(self.vision_api_base)}）",
            f"图片 Key：{self.get_vision_api_key_state()}",
        ]
        if old_provider != new_provider:
            lines.append(
                f"提示：已从 {old_provider} 切换到 {new_provider}。"
                f"LLM_API_KEY 必须属于 {new_provider}；如果仍是旧供应商的 key，下一次请求会鉴权失败。"
            )
        lines.extend(self.get_compatibility_notes())
        return True, "\n".join(lines)

    def get_current_model(self) -> str:
        return self.model

    def get_current_vision_model(self) -> str:
        return self.vision_model

    def get_current_vision_api_base(self) -> str:
        return self.vision_api_base

    def get_current_vision_api_key(self) -> Optional[str]:
        return self.vision_api_key

    def get_current_api_base(self) -> str:
        return self.api_base

    def get_api_provider(self) -> str:
        return provider_from_base_url(self.api_base)

    def get_vision_api_provider(self) -> str:
        return provider_from_base_url(self.vision_api_base)

    def get_api_key_state(self) -> str:
        return format_api_key_state(self.api_key)

    def get_vision_api_key_state(self) -> str:
        state = format_api_key_state(self.vision_api_key)
        if not self._vision_api_key_from_env:
            return f"沿用 LLM_API_KEY（{state}）"
        return state

    def get_compatibility_notes(self) -> list[str]:
        notes = []
        if is_placeholder_api_key(self.api_key):
            notes.append("提示：LLM_API_KEY 缺失或仍是占位值，文字请求不会成功。")
        if self._vision_api_key_from_env and is_placeholder_api_key(self.vision_api_key):
            notes.append("提示：LLM_VISION_API_KEY 缺失或仍是占位值，图片请求不会成功。")
        if (
            not self._vision_api_key_from_env
            and self.get_api_provider() != self.get_vision_api_provider()
        ):
            notes.append(
                "提示：图片模型未单独配置 LLM_VISION_API_KEY，当前会沿用文字 Key；"
                "文字与图片 API Base 属于不同供应商时通常不兼容。"
            )
        return notes

    def list_models(self) -> list[str]:
        return list(SUPPORTED_MODELS.keys())


model_config = ModelConfig()
