"""LangChain chat model wrapper for DeepSeek Chat."""
from __future__ import annotations

import json
import os
import asyncio
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class DeepSeekChat(BaseChatModel):
    """Chat model wrapper for the DeepSeek chat completion API."""

    model_config = {"extra": "allow"}

    model: str = "deepseek-chat"
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com/v1"
    temperature: float = 0.1
    timeout: float = 60.0
    stream: bool = False

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"

    @property
    def _headers(self) -> Dict[str, str]:
        key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek chat.")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        converted: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                role = "assistant"
            converted.append({"role": role, "content": message.content})
        return converted

    def _create_payload(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, stream: bool = False
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        return payload

    def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        base_url = os.getenv("DEEPSEEK_BASE_URL", self.base_url)
        url = f"{base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=self._headers, json=payload)
            response.raise_for_status()
            return response

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._create_payload(messages, stop, stream=False)
        response = self._post(payload)
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        generation_info = {"finish_reason": choice.get("finish_reason"), "id": data.get("id")}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content), generation_info=generation_info)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterable[ChatGenerationChunk]:
        payload = self._create_payload(messages, stop, stream=True)
        base_url = os.getenv("DEEPSEEK_BASE_URL", self.base_url)
        url = f"{base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, headers=self._headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data:"):
                        content = line[len(b"data:") :].strip()
                        if content == b"[DONE]":
                            break
                        chunk_payload = json.loads(content)
                        choice = chunk_payload.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=text),
                                generation_info={"finish_reason": choice.get("finish_reason"), "id": chunk_payload.get("id")},
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(text, chunk=chunk)
                            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterable[ChatGenerationChunk]:
        loop = asyncio.get_event_loop()

        def sync_stream() -> Iterable[ChatGenerationChunk]:
            return self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

        queue: asyncio.Queue[Optional[ChatGenerationChunk]] = asyncio.Queue()

        def produce() -> None:
            try:
                for chunk in sync_stream():
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer = loop.run_in_executor(None, produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await producer
