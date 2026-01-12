"""Database layer for logging and retrieving LLM requests."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import mysql.connector
from mysql.connector import MySQLConnection
from mysql.connector.cursor import MySQLCursor
from ruamel.yaml import YAML


@dataclass
class DBConfig:
    """MySQL connection configuration."""
    host: str = "localhost"
    port: int = 3306
    database: str = "vllm"
    user: str = "vllm_user"
    password: str = "vllm_pass"

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DBConfig":
        """Load database config from config.yaml."""
        yaml = YAML()
        config = yaml.load(Path(config_path).read_text())
        db_cfg = config.get("database", {}).get("mysql", {})
        return cls(
            host=db_cfg.get("host", "localhost"),
            port=int(db_cfg.get("port", 3306)),
            database=db_cfg.get("database", "vllm"),
            user=db_cfg.get("user", "vllm_user"),
            password=db_cfg.get("password", "vllm_pass"),
        )


@dataclass
class Database:
    """MySQL database connection and operations."""
    
    config: DBConfig
    _connection: MySQLConnection | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls, config_path: str | Path = "infra/config.yaml") -> "Database":
        """Create database from config file."""
        return cls(config=DBConfig.from_config(config_path))

    def connect(self) -> MySQLConnection:
        """Get or create database connection."""
        if self._connection is None or not self._connection.is_connected():
            self._connection = mysql.connector.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                autocommit=True,
            )
        return self._connection

    def close(self) -> None:
        """Close database connection."""
        if self._connection and self._connection.is_connected():
            self._connection.close()
            self._connection = None

    @contextmanager
    def cursor(self, dictionary: bool = True) -> Generator[MySQLCursor, None, None]:
        """Get a database cursor with automatic cleanup."""
        conn = self.connect()
        cursor = conn.cursor(dictionary=dictionary)
        try:
            yield cursor
        finally:
            cursor.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Prompts (Immutable)
    # ─────────────────────────────────────────────────────────────────────────

    def get_or_create_prompt(
        self,
        content: str,
        name: str | None = None,
        description: str | None = None,
    ) -> int:
        """
        Get or create an immutable prompt.
        
        If a prompt with the same content already exists (by hash), returns its ID.
        Otherwise creates a new prompt.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        with self.cursor() as cursor:
            # Check if exists
            cursor.execute(
                "SELECT id FROM prompts WHERE content_hash = %s",
                (content_hash,)
            )
            row = cursor.fetchone()
            if row:
                return row["id"]
            
            # Create new
            cursor.execute(
                """
                INSERT INTO prompts (content_hash, content, name, description)
                VALUES (%s, %s, %s, %s)
                """,
                (content_hash, content, name, description)
            )
            return cursor.lastrowid

    def get_prompt(self, prompt_id: int) -> dict[str, Any] | None:
        """Get a prompt by ID."""
        with self.cursor() as cursor:
            cursor.execute("SELECT * FROM prompts WHERE id = %s", (prompt_id,))
            return cursor.fetchone()

    # ─────────────────────────────────────────────────────────────────────────
    # Models
    # ─────────────────────────────────────────────────────────────────────────

    def get_or_create_model(
        self,
        name: str,
        model_type: str = "chat",
        description: str | None = None,
        provider: str = "ollama",
    ) -> int:
        """Get or create a model entry."""
        with self.cursor() as cursor:
            cursor.execute("SELECT id FROM models WHERE name = %s", (name,))
            row = cursor.fetchone()
            if row:
                return row["id"]
            
            cursor.execute(
                """
                INSERT INTO models (name, model_type, description, provider)
                VALUES (%s, %s, %s, %s)
                """,
                (name, model_type, description, provider)
            )
            return cursor.lastrowid

    def get_model(self, model_id: int) -> dict[str, Any] | None:
        """Get a model by ID."""
        with self.cursor() as cursor:
            cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
            return cursor.fetchone()

    def get_model_by_name(self, name: str) -> dict[str, Any] | None:
        """Get a model by name."""
        with self.cursor() as cursor:
            cursor.execute("SELECT * FROM models WHERE name = %s", (name,))
            return cursor.fetchone()

    # ─────────────────────────────────────────────────────────────────────────
    # Chat Requests
    # ─────────────────────────────────────────────────────────────────────────

    def log_chat_request(
        self,
        model_name: str,
        user_message: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        extra_params: dict | None = None,
    ) -> str:
        """
        Log a chat request before sending.
        
        Returns the request_id (UUID) for later updating with response.
        """
        request_id = str(uuid.uuid4())
        
        # Get or create model
        model_id = self.get_or_create_model(model_name, "chat")
        
        # Get or create system prompt if provided
        system_prompt_id = None
        if system_prompt:
            system_prompt_id = self.get_or_create_prompt(system_prompt)
        
        with self.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_requests (
                    request_id, model_id, model_name, system_prompt_id,
                    user_message, messages_json, temperature, max_tokens,
                    extra_params, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                """,
                (
                    request_id, model_id, model_name, system_prompt_id,
                    user_message, json.dumps(messages), temperature, max_tokens,
                    json.dumps(extra_params) if extra_params else None,
                )
            )
        
        return request_id

    def update_chat_response(
        self,
        request_id: str,
        response_content: str,
        response_json: dict,
        response_time_ms: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """
        Update a chat request with the response.
        
        Args:
            request_id: The request UUID
            response_content: The final answer/content from the model
            response_json: Full API response for reproducibility
            response_time_ms: How long the request took
            prompt_tokens: Input tokens used
            completion_tokens: Output tokens generated
            reasoning_content: Chain-of-thought reasoning (for thinking models)
        """
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        
        with self.cursor() as cursor:
            cursor.execute(
                """
                UPDATE chat_requests SET
                    response_content = %s,
                    reasoning_content = %s,
                    response_json = %s,
                    response_time_ms = %s,
                    prompt_tokens = %s,
                    completion_tokens = %s,
                    total_tokens = %s,
                    status = 'success',
                    completed_at = NOW(3)
                WHERE request_id = %s
                """,
                (
                    response_content, reasoning_content, json.dumps(response_json),
                    response_time_ms, prompt_tokens, completion_tokens,
                    total_tokens, request_id,
                )
            )

    def update_chat_error(self, request_id: str, error_message: str) -> None:
        """Update a chat request with an error."""
        with self.cursor() as cursor:
            cursor.execute(
                """
                UPDATE chat_requests SET
                    status = 'error',
                    error_message = %s,
                    completed_at = NOW(3)
                WHERE request_id = %s
                """,
                (error_message, request_id)
            )

    def get_chat_request(self, request_id: str) -> dict[str, Any] | None:
        """Get a chat request by ID."""
        with self.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM chat_requests WHERE request_id = %s",
                (request_id,)
            )
            return cursor.fetchone()

    def get_recent_chat_requests(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent chat requests."""
        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM chat_requests
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            return cursor.fetchall()

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Requests
    # ─────────────────────────────────────────────────────────────────────────

    def log_embedding_request(
        self,
        model_name: str,
        input_text: str | list[str],
    ) -> str:
        """
        Log an embedding request before sending.
        
        Returns the request_id (UUID) for later updating with response.
        """
        request_id = str(uuid.uuid4())
        
        # Normalize input
        if isinstance(input_text, str):
            input_list = [input_text]
            primary_text = input_text
        else:
            input_list = list(input_text)
            primary_text = input_list[0] if input_list else ""
        
        # Get or create model
        model_id = self.get_or_create_model(model_name, "embedding")
        
        with self.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_requests (
                    request_id, model_id, model_name,
                    input_text, input_json, input_count, status
                ) VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                """,
                (
                    request_id, model_id, model_name,
                    primary_text, json.dumps(input_list), len(input_list),
                )
            )
        
        return request_id

    def update_embedding_response(
        self,
        request_id: str,
        embeddings: list[list[float]],
        response_time_ms: int,
        total_tokens: int | None = None,
    ) -> None:
        """Update an embedding request with the response."""
        dimensions = len(embeddings[0]) if embeddings else None
        
        with self.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_requests SET
                    dimensions = %s,
                    embeddings_json = %s,
                    response_time_ms = %s,
                    total_tokens = %s,
                    status = 'success',
                    completed_at = NOW(3)
                WHERE request_id = %s
                """,
                (
                    dimensions, json.dumps(embeddings),
                    response_time_ms, total_tokens, request_id,
                )
            )

    def update_embedding_error(self, request_id: str, error_message: str) -> None:
        """Update an embedding request with an error."""
        with self.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_requests SET
                    status = 'error',
                    error_message = %s,
                    completed_at = NOW(3)
                WHERE request_id = %s
                """,
                (error_message, request_id)
            )

    def get_embedding_request(self, request_id: str) -> dict[str, Any] | None:
        """Get an embedding request by ID."""
        with self.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM embedding_requests WHERE request_id = %s",
                (request_id,)
            )
            return cursor.fetchone()

    def get_recent_embedding_requests(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent embedding requests."""
        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM embedding_requests
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            return cursor.fetchall()

    # ─────────────────────────────────────────────────────────────────────────
    # Utility Queries
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_models(self) -> list[dict[str, Any]]:
        """Get all registered models."""
        with self.cursor() as cursor:
            cursor.execute("SELECT * FROM models ORDER BY model_type, name")
            return cursor.fetchall()

    def get_request_stats(self) -> dict[str, Any]:
        """Get summary statistics of all requests."""
        with self.cursor() as cursor:
            # Chat stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                    AVG(response_time_ms) as avg_time_ms,
                    SUM(total_tokens) as total_tokens
                FROM chat_requests
            """)
            chat_stats = cursor.fetchone()
            
            # Embedding stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors,
                    AVG(response_time_ms) as avg_time_ms,
                    SUM(input_count) as total_embeddings
                FROM embedding_requests
            """)
            embed_stats = cursor.fetchone()
            
            return {
                "chat": chat_stats,
                "embedding": embed_stats,
            }
