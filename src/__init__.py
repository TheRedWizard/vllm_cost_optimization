"""vLLM Cost Optimization - Ollama client utilities."""

from .db import Database, DBConfig
from .ollama_client import (
    OllamaClient,
    call_embedding,
    call_model,
    get_chat_models,
    get_embedding_models,
)
from .tools import (
    ToolRegistry,
    ToolResult,
    call_tool,
    init_registry,
    list_tools,
)

__all__ = [
    # Ollama client
    "OllamaClient",
    "call_model",
    "call_embedding",
    "get_chat_models",
    "get_embedding_models",
    # Database
    "Database",
    "DBConfig",
    # Tools
    "ToolRegistry",
    "ToolResult",
    "call_tool",
    "init_registry",
    "list_tools",
]
