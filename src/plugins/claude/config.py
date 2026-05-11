"""模型切换配置模块 - 支持运行时切换模型"""

import os
from typing import Optional

# 支持的模型列表
SUPPORTED_MODELS = {
    "qwen-chat": "https://api.llm.ustc.edu.cn/v1",
    "qwen2.5-72b": "https://api.llm.ustc.edu.cn/v1",
    "deepseek-v4-flash": "https://api.llm.ustc.edu.cn/v1",
    "deepseek-v4-pro": "https://api.llm.ustc.edu.cn/v1",
    "deepseek-chat": "https://api.deepseek.com/v1",
}

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

        self.model = model_name
        self.api_base = SUPPORTED_MODELS[model_name]
        os.environ["LLM_MODEL"] = model_name
        os.environ["LLM_API_BASE"] = self.api_base
        return True, f"已切换到模型：{model_name}"

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

    def list_models(self) -> list[str]:
        return list(SUPPORTED_MODELS.keys())


model_config = ModelConfig()
