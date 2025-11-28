"""Embeddings wrapper for SiliconFlow's BGE3 endpoint."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.embeddings import Embeddings


class SiliconFlowEmbeddings(Embeddings):
    """LangChain embeddings integration for SiliconFlow BGE models."""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        dimensions: Optional[int] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is required for SiliconFlow embeddings.")
        resolved_base = base_url or os.getenv("SILICONFLOW_BASE_URL")
        self.base_url = (resolved_base or "https://api.siliconflow.cn/v1/embeddings").rstrip("/")
        self.timeout = timeout
        self.dimensions = dimensions

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_payload(self, texts: List[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"input": texts, "model": self.model}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return payload

    def _post_embeddings(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.base_url,
                headers=self._headers,
                json=self._request_payload(texts),
            )
            response.raise_for_status()
        data = response.json()
        embeddings = [item.get("embedding") for item in data.get("data", [])]
        if not embeddings:
            raise ValueError("No embeddings returned from SiliconFlow response.")
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._post_embeddings([text])[0]
