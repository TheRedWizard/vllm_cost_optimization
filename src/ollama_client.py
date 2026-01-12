"""Ollama client for calling models via OpenAI-compatible API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from ruamel.yaml import YAML

try:
    from .db import Database
except ImportError:
    from db import Database


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    thinking: bool = False
    thinking_budget: int = 2048  # Extra tokens for reasoning models
    
    @classmethod
    def from_dict(cls, data: dict | str) -> "ModelConfig":
        """Parse model config from dict or string."""
        if isinstance(data, str):
            return cls(name=data)
        return cls(
            name=data["name"],
            thinking=data.get("thinking", False),
            thinking_budget=data.get("thinking_budget", 2048),
        )


@dataclass
class ChatResponse:
    """Structured response from a chat completion."""
    content: str
    reasoning: str | None = None
    raw: dict = field(default_factory=dict)
    
    @property
    def full_response(self) -> str:
        """Get the complete response (reasoning + content if both present)."""
        if self.reasoning and self.content:
            return f"<thinking>\n{self.reasoning}\n</thinking>\n\n{self.content}"
        elif self.reasoning:
            return f"<thinking>\n{self.reasoning}\n</thinking>"
        return self.content or ""
    
    @property
    def has_reasoning(self) -> bool:
        """Check if response includes chain-of-thought reasoning."""
        return self.reasoning is not None and len(self.reasoning) > 0
    
    @property
    def is_truncated(self) -> bool:
        """Check if the response was likely truncated (reasoning but no content)."""
        return self.has_reasoning and not self.content
    
    @classmethod
    def from_api_response(cls, data: dict) -> "ChatResponse":
        """Parse response from Ollama API."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        content = message.get("content", "")
        reasoning = message.get("reasoning")  # DeepSeek-R1 / Qwen3 style
        
        # Keep content and reasoning separate - don't mix them
        # If content is empty but reasoning exists, the model was truncated
        # Store both as-is and let the caller decide how to handle
        
        return cls(content=content, reasoning=reasoning, raw=data)


@dataclass
class OllamaClient:
    """Client for interacting with Ollama's OpenAI-compatible API."""

    base_url: str
    api_key: str = "none"
    timeout: int = 120
    model_configs: dict[str, ModelConfig] = field(default_factory=dict)
    db: Database | None = field(default=None, repr=False)
    log_to_db: bool = True

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "infra/config.yaml",
        log_to_db: bool = True,
    ) -> "OllamaClient":
        """Create client from config.yaml file."""
        yaml = YAML()
        config = yaml.load(Path(config_path).read_text())
        ollama_cfg = config["endpoints"]["ollama"]
        
        # Parse model configs
        model_configs = {}
        for model_data in ollama_cfg.get("chat_models", []):
            mc = ModelConfig.from_dict(model_data)
            model_configs[mc.name] = mc
        
        db = None
        if log_to_db:
            try:
                db = Database.from_config(config_path)
                db.connect()
            except Exception:
                db = None
                log_to_db = False
        
        return cls(
            base_url=ollama_cfg["base_url"].rstrip("/"),
            api_key=ollama_cfg.get("api_key", "none"),
            model_configs=model_configs,
            db=db,
            log_to_db=log_to_db,
        )

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _resolve_max_tokens(
        self,
        model: str,
        max_tokens: int | None,
        output_tokens: int | None = None,
    ) -> int | None:
        """
        Resolve max_tokens for a request.
        
        Logic:
        - If neither specified: return None (model decides - usually best)
        - If output_tokens specified: that's how many output tokens you want
        - For thinking models: add thinking_budget to ensure room for reasoning
        
        Args:
            model: Model name
            max_tokens: Explicit max_tokens (total budget)
            output_tokens: Desired output tokens (we'll add thinking budget)
        """
        model_cfg = self.model_configs.get(model)
        
        # If explicit max_tokens set, use it directly
        if max_tokens is not None:
            return max_tokens
        
        # If output_tokens specified for a thinking model, add budget
        if output_tokens is not None:
            if model_cfg and model_cfg.thinking:
                return output_tokens + model_cfg.thinking_budget
            return output_tokens
        
        # Default: let model decide (best for most cases)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Chat Completions
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Send a chat completion request to Ollama.

        Args:
            model: Model name (e.g., "qwen3:8b")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0-2)
            max_tokens: Hard limit on total tokens (use sparingly)
            output_tokens: Desired output length - thinking models get extra budget
            **kwargs: Additional parameters to pass to the API

        Returns:
            ChatResponse with content, reasoning (if present), and raw response
        """
        url = f"{self.base_url}/chat/completions"
        
        # Resolve max_tokens intelligently
        resolved_max_tokens = self._resolve_max_tokens(model, max_tokens, output_tokens)
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if resolved_max_tokens is not None:
            payload["max_tokens"] = resolved_max_tokens
        payload.update(kwargs)

        # Extract user message and system prompt for logging
        user_message = ""
        system_prompt = None
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
            elif msg.get("role") == "system":
                system_prompt = msg.get("content")

        # Log request to database
        request_id = None
        if self.log_to_db and self.db:
            try:
                request_id = self.db.log_chat_request(
                    model_name=model,
                    user_message=user_message,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=resolved_max_tokens,
                    system_prompt=system_prompt,
                    extra_params=kwargs if kwargs else None,
                )
            except Exception:
                request_id = None

        # Make request
        start_time = time.time()
        try:
            response = requests.post(
                url, json=payload, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Parse response
            chat_response = ChatResponse.from_api_response(result)

            # Update database with response (content and reasoning stored separately)
            if request_id and self.db:
                try:
                    usage = result.get("usage", {})
                    self.db.update_chat_response(
                        request_id=request_id,
                        response_content=chat_response.content,
                        response_json=result,
                        response_time_ms=elapsed_ms,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        reasoning_content=chat_response.reasoning,
                    )
                except Exception:
                    pass

            return chat_response

        except Exception as e:
            if request_id and self.db:
                try:
                    self.db.update_chat_error(request_id, str(e))
                except Exception:
                    pass
            raise

    def generate(self, model: str, prompt: str, **kwargs: Any) -> str:
        """
        Simple generation helper - returns just the text response.

        Args:
            model: Model name
            prompt: User prompt
            **kwargs: Additional parameters (output_tokens recommended over max_tokens)

        Returns:
            Generated text content (includes reasoning for thinking models)
        """
        messages = [{"role": "user", "content": prompt}]
        result = self.chat(model, messages, **kwargs)
        return result.content

    # ─────────────────────────────────────────────────────────────────────────
    # Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    def embeddings(
        self,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate embeddings for text input.

        Args:
            model: Embedding model name (e.g., "nomic-embed-text:latest")
            input: Single text string or list of strings to embed
            **kwargs: Additional parameters

        Returns:
            Full API response with embeddings
        """
        url = f"{self.base_url}/embeddings"
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        payload.update(kwargs)

        # Log request to database
        request_id = None
        if self.log_to_db and self.db:
            try:
                request_id = self.db.log_embedding_request(
                    model_name=model,
                    input_text=input,
                )
            except Exception:
                request_id = None

        # Make request
        start_time = time.time()
        try:
            response = requests.post(
                url, json=payload, headers=self._headers(), timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Update database with response
            if request_id and self.db:
                try:
                    embeddings = [item["embedding"] for item in result.get("data", [])]
                    usage = result.get("usage", {})
                    self.db.update_embedding_response(
                        request_id=request_id,
                        embeddings=embeddings,
                        response_time_ms=elapsed_ms,
                        total_tokens=usage.get("total_tokens"),
                    )
                except Exception:
                    pass

            return result

        except Exception as e:
            if request_id and self.db:
                try:
                    self.db.update_embedding_error(request_id, str(e))
                except Exception:
                    pass
            raise

    def embed(self, model: str, text: str | list[str], **kwargs: Any) -> list[list[float]]:
        """
        Simple embedding helper - returns just the embedding vectors.

        Args:
            model: Embedding model name
            text: Text or list of texts to embed
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors (list of floats)
        """
        result = self.embeddings(model, text, **kwargs)
        return [item["embedding"] for item in result["data"]]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def call_model(
    model: str,
    prompt: str,
    config_path: str | Path = "infra/config.yaml",
    log_to_db: bool = True,
    **kwargs: Any,
) -> str:
    """
    Convenience function to call an Ollama chat model.

    Args:
        model: Model name (e.g., "qwen3:8b")
        prompt: The prompt to send
        config_path: Path to config.yaml
        log_to_db: Whether to log the request to MySQL
        **kwargs: Additional parameters (output_tokens recommended over max_tokens)

    Returns:
        Generated text response
    """
    client = OllamaClient.from_config(config_path, log_to_db=log_to_db)
    return client.generate(model, prompt, **kwargs)


def call_embedding(
    model: str,
    text: str | list[str],
    config_path: str | Path = "infra/config.yaml",
    log_to_db: bool = True,
    **kwargs: Any,
) -> list[list[float]]:
    """
    Convenience function to call an Ollama embedding model.

    Args:
        model: Embedding model name (e.g., "nomic-embed-text:latest")
        text: Text or list of texts to embed
        config_path: Path to config.yaml
        log_to_db: Whether to log the request to MySQL
        **kwargs: Additional parameters

    Returns:
        List of embedding vectors
    """
    client = OllamaClient.from_config(config_path, log_to_db=log_to_db)
    return client.embed(model, text, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Config Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_chat_models(config_path: str | Path = "infra/config.yaml") -> list[str]:
    """Get list of available chat model names from config."""
    yaml = YAML()
    config = yaml.load(Path(config_path).read_text())
    models = []
    for m in config["endpoints"]["ollama"].get("chat_models", []):
        if isinstance(m, str):
            models.append(m)
        else:
            models.append(m["name"])
    return models


def get_embedding_models(config_path: str | Path = "infra/config.yaml") -> list[str]:
    """Get list of available embedding model names from config."""
    yaml = YAML()
    config = yaml.load(Path(config_path).read_text())
    models = []
    for m in config["endpoints"]["ollama"].get("embedding_models", []):
        if isinstance(m, str):
            models.append(m)
        else:
            models.append(m["name"])
    return models
