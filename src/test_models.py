#!/usr/bin/env python3
"""Test all configured Ollama models with simple prompts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent))

from db import Database
from ollama_client import (
    OllamaClient,
    get_chat_models,
    get_embedding_models,
)


def test_chat_models(client: OllamaClient, models: list[str]) -> tuple[int, int]:
    """Test chat models with a simple prompt."""
    print("\n" + "=" * 60)
    print("CHAT MODELS")
    print("=" * 60)

    test_prompt = "Say 'Hello!' and nothing else."
    passed = 0
    failed = 0

    for model in models:
        print(f"\n{'─' * 60}")
        print(f"Testing: {model}")
        
        # Show if it's a thinking model
        model_cfg = client.model_configs.get(model)
        if model_cfg and model_cfg.thinking:
            print(f"  [thinking model, budget: {model_cfg.thinking_budget}]")
        print(f"{'─' * 60}")

        start = time.time()
        try:
            # Use output_tokens for a reasonable limit
            response = client.chat(model, [{"role": "user", "content": test_prompt}], output_tokens=100)
            elapsed = time.time() - start

            print(f"  ✓ Response: {response.content.strip()[:100]}")
            if response.reasoning:
                print(f"  ✓ Reasoning: {response.reasoning[:60]}...")
            print(f"  ✓ Time: {elapsed:.2f}s")
            passed += 1

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ Error: {e}")
            print(f"  ✗ Time: {elapsed:.2f}s")
            failed += 1

    return passed, failed


def test_embedding_models(client: OllamaClient, models: list[str]) -> tuple[int, int]:
    """Test embedding models with sample text."""
    print("\n" + "=" * 60)
    print("EMBEDDING MODELS")
    print("=" * 60)

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    passed = 0
    failed = 0

    for model in models:
        print(f"\n{'─' * 60}")
        print(f"Testing: {model}")
        print(f"{'─' * 60}")

        start = time.time()
        try:
            embeddings = client.embed(model, test_texts)
            elapsed = time.time() - start

            # Validate embeddings
            assert len(embeddings) == len(test_texts), "Wrong number of embeddings returned"
            dims = [len(emb) for emb in embeddings]

            print(f"  ✓ Embeddings: {len(embeddings)} vectors")
            print(f"  ✓ Dimensions: {dims[0]}")
            print(f"  ✓ Time: {elapsed:.2f}s")
            passed += 1

        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ Error: {e}")
            print(f"  ✗ Time: {elapsed:.2f}s")
            failed += 1

    return passed, failed


def show_database_results(config_path: Path) -> None:
    """Show what was logged to the database."""
    print("\n" + "=" * 60)
    print("DATABASE RESULTS")
    print("=" * 60)

    try:
        db = Database.from_config(config_path)
        
        # Get stats
        stats = db.get_request_stats()
        print("\n📊 Request Statistics:")
        print(f"  Chat Requests:      {stats['chat']['total']} total, {stats['chat']['success']} success")
        print(f"  Embedding Requests: {stats['embedding']['total']} total, {stats['embedding']['success']} success")
        
        if stats['chat']['avg_time_ms']:
            print(f"  Avg Chat Time:      {stats['chat']['avg_time_ms']:.0f}ms")
        if stats['embedding']['avg_time_ms']:
            print(f"  Avg Embed Time:     {stats['embedding']['avg_time_ms']:.0f}ms")
        
        # Recent chat requests
        print("\n📝 Recent Chat Requests:")
        chat_requests = db.get_recent_chat_requests(5)
        for req in chat_requests:
            status_icon = "✓" if req["status"] == "success" else "✗"
            print(f"  {status_icon} [{req['model_name']}] {req['user_message'][:40]}...")
            if req["response_content"]:
                # Handle multi-line responses
                first_line = req["response_content"].split('\n')[0][:50]
                print(f"    → {first_line}...")
            print(f"    Time: {req['response_time_ms']}ms | ID: {req['request_id'][:8]}...")
        
        # Recent embedding requests
        print("\n🔢 Recent Embedding Requests:")
        embed_requests = db.get_recent_embedding_requests(5)
        for req in embed_requests:
            status_icon = "✓" if req["status"] == "success" else "✗"
            print(f"  {status_icon} [{req['model_name']}] {req['input_text'][:40]}...")
            if req["dimensions"]:
                print(f"    → {req['input_count']} vectors, {req['dimensions']} dims")
            print(f"    Time: {req['response_time_ms']}ms | ID: {req['request_id'][:8]}...")
        
        db.close()
        
    except Exception as e:
        print(f"\n⚠ Could not connect to database: {e}")
        print("  Run: cd infra && ./setup_mysql.sh")


def test_all_models(config_path: str | Path = "infra/config.yaml") -> int:
    """Test all models from config."""
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "infra" / "config.yaml"

    print(f"Loading config from: {config_path}")

    chat_models = get_chat_models(config_path)
    embedding_models = get_embedding_models(config_path)

    print(f"Found {len(chat_models)} chat models")
    print(f"Found {len(embedding_models)} embedding models")

    if not chat_models and not embedding_models:
        print("No models found in config!")
        return 1

    client = OllamaClient.from_config(config_path, log_to_db=True)
    
    if client.db:
        print("✓ Database logging enabled")
    else:
        print("⚠ Database logging disabled (MySQL not available)")
    
    # Show thinking model info
    thinking_models = [name for name, cfg in client.model_configs.items() if cfg.thinking]
    if thinking_models:
        print(f"✓ Thinking models: {', '.join(thinking_models)}")

    # Test chat models
    chat_passed, chat_failed = 0, 0
    if chat_models:
        chat_passed, chat_failed = test_chat_models(client, chat_models)

    # Test embedding models
    embed_passed, embed_failed = 0, 0
    if embedding_models:
        embed_passed, embed_failed = test_embedding_models(client, embedding_models)

    # Summary
    total_passed = chat_passed + embed_passed
    total_failed = chat_failed + embed_failed
    total = total_passed + total_failed

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Chat Models:      {chat_passed}/{chat_passed + chat_failed} passed")
    print(f"Embedding Models: {embed_passed}/{embed_passed + embed_failed} passed")
    print(f"{'─' * 60}")
    print(f"Total:            {total_passed}/{total} passed")

    # Show database results
    show_database_results(config_path)

    if total_failed > 0:
        print(f"\n⚠ {total_failed} model(s) failed!")
        return 1

    print("\n✓ All models passed!")
    return 0


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "infra/config.yaml"
    raise SystemExit(test_all_models(config))
