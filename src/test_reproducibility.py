#!/usr/bin/env python3
"""
Test reproducibility of LLM outputs.

Verifies that the same input with temperature=0 produces identical outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from db import Database
from ollama_client import OllamaClient, get_chat_models


def test_reproducibility(
    config_path: str = "infra/config.yaml",
    num_runs: int = 3,
    test_prompt: str = "What is 2+2? Answer with just the number.",
) -> int:
    """
    Test that identical inputs produce identical outputs.
    
    Args:
        config_path: Path to config file
        num_runs: Number of times to run each model
        test_prompt: The prompt to test with
    """
    print("=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)
    print(f"Prompt: {test_prompt}")
    print(f"Runs per model: {num_runs}")
    print(f"Temperature: 0 (deterministic)")
    print()
    
    client = OllamaClient.from_config(config_path, log_to_db=True)
    db = Database.from_config(config_path)
    models = get_chat_models(config_path)
    
    results = {}
    all_passed = True
    
    for model in models:
        print(f"{'─' * 70}")
        print(f"Testing: {model}")
        print(f"{'─' * 70}")
        
        responses = []
        reasonings = []
        request_ids = []
        
        for run in range(num_runs):
            resp = client.chat(
                model,
                [{"role": "user", "content": test_prompt}],
                temperature=0,  # Deterministic
                seed=42,        # Fixed seed for reproducibility
            )
            responses.append(resp.content.strip())
            reasonings.append(resp.reasoning.strip() if resp.reasoning else None)
            
            # Get the request ID from the database (most recent)
            recent = db.get_recent_chat_requests(1)
            if recent:
                request_ids.append(recent[0]["request_id"])
            
            print(f"  Run {run + 1}: {resp.content.strip()[:60]}...")
        
        # Check if all responses are identical
        unique_responses = set(responses)
        unique_reasonings = set(r for r in reasonings if r is not None)
        
        content_match = len(unique_responses) == 1
        reasoning_match = len(unique_reasonings) <= 1  # All same or all None
        
        results[model] = {
            "responses": responses,
            "reasonings": reasonings,
            "request_ids": request_ids,
            "content_match": content_match,
            "reasoning_match": reasoning_match,
        }
        
        if content_match and reasoning_match:
            print(f"  ✓ REPRODUCIBLE - All {num_runs} runs identical")
        else:
            print(f"  ✗ NOT REPRODUCIBLE")
            if not content_match:
                print(f"    Content variations: {unique_responses}")
            if not reasoning_match:
                print(f"    Reasoning variations: {len(unique_reasonings)} different")
            all_passed = False
        
        print()
    
    # Now test replaying from database
    print("=" * 70)
    print("DATABASE REPLAY TEST")
    print("=" * 70)
    print("Re-running requests from database to verify reproducibility...")
    print()
    
    replay_passed = True
    
    for model, data in results.items():
        if not data["request_ids"]:
            continue
            
        # Get the first request from database
        request_id = data["request_ids"][0]
        original = db.get_chat_request(request_id)
        
        if not original:
            print(f"  ⚠ Could not find request {request_id}")
            continue
        
        print(f"{'─' * 70}")
        print(f"Replaying: {model}")
        print(f"Original request ID: {request_id[:8]}...")
        print(f"{'─' * 70}")
        
        # Parse original request parameters
        messages = json.loads(original["messages_json"]) if isinstance(original["messages_json"], str) else original["messages_json"]
        
        # Replay with same parameters
        resp = client.chat(
            model,
            messages,
            temperature=0,
            seed=42,
        )
        
        original_content = original["response_content"].strip() if original["response_content"] else ""
        replay_content = resp.content.strip()
        
        original_reasoning = original["reasoning_content"].strip() if original.get("reasoning_content") else None
        replay_reasoning = resp.reasoning.strip() if resp.reasoning else None
        
        content_match = original_content == replay_content
        reasoning_match = original_reasoning == replay_reasoning
        
        print(f"  Original content:  {original_content[:50]}...")
        print(f"  Replayed content:  {replay_content[:50]}...")
        
        if content_match:
            print(f"  ✓ Content matches")
        else:
            print(f"  ✗ Content differs!")
            replay_passed = False
        
        if original_reasoning or replay_reasoning:
            if reasoning_match:
                print(f"  ✓ Reasoning matches")
            else:
                print(f"  ✗ Reasoning differs!")
                replay_passed = False
        
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed_models = sum(1 for r in results.values() if r["content_match"] and r["reasoning_match"])
    total_models = len(results)
    
    print(f"Same-session reproducibility: {passed_models}/{total_models} models")
    print(f"Database replay: {'PASSED' if replay_passed else 'FAILED'}")
    print()
    
    if all_passed and replay_passed:
        print("✓ All reproducibility tests passed!")
        print()
        print("NOTE: Reproducibility with temperature=0 works because:")
        print("  - No random sampling (greedy decoding)")
        print("  - Fixed seed ensures deterministic behavior")
        print("  - Same model version and parameters")
        return 0
    else:
        print("⚠ Some tests failed.")
        print()
        print("Possible causes of non-reproducibility:")
        print("  - Model doesn't fully support seed parameter")
        print("  - Floating point variations across runs")
        print("  - Model was updated between runs")
        return 1


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "infra/config.yaml"
    raise SystemExit(test_reproducibility(config))
