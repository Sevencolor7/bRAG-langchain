"""Custom integrations for third-party model providers."""

from .siliconflow import SiliconFlowEmbeddings
from .deepseek import DeepSeekChat

__all__ = ["SiliconFlowEmbeddings", "DeepSeekChat"]
