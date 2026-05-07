"""LLM API 调用模块 - 支持 OpenAI 兼容格式"""

import os
import httpx
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
_project_root = Path(__file__).resolve().parents[3]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

class LLMClient:
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
    ):
        self.base_url = base_url or os.getenv("LLM_API_BASE", "https://api.llm.ustc.edu.cn/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "qwen-chat")

        self.client = self._build_client()

    def _build_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(60.0, connect=15.0),
            ),
        )

    def configure(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Update runtime API configuration.

        The OpenAI client must be rebuilt when endpoint/key changes; model changes
        can be applied directly because each request passes the current model.
        """
        rebuild_client = False

        if base_url is not None and base_url != self.base_url:
            self.base_url = base_url
            rebuild_client = True

        if api_key is not None and api_key != self.api_key:
            self.api_key = api_key
            rebuild_client = True

        if model is not None:
            self.model = model

        if rebuild_client:
            self.client = self._build_client()

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """发送聊天请求并返回回复"""

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    async def chat_with_image(
        self,
        messages: List[Dict[str, Any]],
        image_base64: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """发送带图片的聊天请求"""

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        # 构建含图片的消息
        content = []
        for msg in messages:
            if msg.get("image"):
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{msg['image']}"}})
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})

        full_messages.append({"role": "user", "content": content})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=2048,
        )

        return response.choices[0].message.content


# 全局单例
llm_client = LLMClient()
