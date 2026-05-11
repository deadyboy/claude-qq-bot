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
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-pro")

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

    def _needs_temporary_client(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        return bool(
            (base_url is not None and base_url != self.base_url)
            or (api_key is not None and api_key != self.api_key)
        )

    async def _create_chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Call the configured endpoint, or a temporary endpoint override.

        Vision and text models may live on different OpenAI-compatible bases.
        Keeping the override local avoids mutating the global text client when
        a single image message needs a dedicated multimodal model.
        """
        if not self._needs_temporary_client(base_url=base_url, api_key=api_key):
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        temp_client = AsyncOpenAI(
            base_url=base_url or self.base_url,
            api_key=api_key or self.api_key,
            http_client=httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(60.0, connect=15.0),
            ),
        )
        try:
            return await temp_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        finally:
            await temp_client.close()

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """发送聊天请求并返回回复"""

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        response = await self._create_chat_completion(
            model=model or self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

        return response.choices[0].message.content

    async def chat_with_images(
        self,
        messages: List[Dict[str, Any]],
        image_urls: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """发送带图片 URL 的聊天请求，图片只附加到最新用户消息。"""

        full_messages: List[Dict[str, Any]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        history = list(messages[:-1]) if messages else []
        for msg in history:
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                full_messages.append({"role": role, "content": content})

        latest_content = ""
        if messages:
            latest_content = str(messages[-1].get("content") or "").strip()
        if not latest_content:
            latest_content = "请描述这张图片。"

        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": latest_content}]
        for url in image_urls:
            clean_url = str(url or "").strip()
            if clean_url:
                content_parts.append({"type": "image_url", "image_url": {"url": clean_url}})

        full_messages.append({"role": "user", "content": content_parts})

        response = await self._create_chat_completion(
            model=model or self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

        return response.choices[0].message.content

    async def chat_with_image(
        self,
        messages: List[Dict[str, Any]],
        image_base64: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """发送带图片的聊天请求"""
        return await self.chat_with_images(
            messages,
            [f"data:image/jpeg;base64,{image_base64}"],
            system_prompt=system_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )


# 全局单例
llm_client = LLMClient()
