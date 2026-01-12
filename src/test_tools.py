#!/usr/bin/env python3
"""Test external data source tools."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools import init_registry, call_tool, list_tools
from db import Database


def test_tools(config_path: str = "infra/config.yaml") -> int:
    """Test various data source tools."""
    print("=" * 70)
    print("TOOL TESTS")
    print("=" * 70)
    
    # Initialize registry with DB
    registry = init_registry(config_path)
    
    if registry.db:
        print("✓ Database auditing enabled")
    else:
        print("⚠ Database auditing disabled")
    
    # List available tools
    tools = list_tools()
    print(f"\nRegistered {len(tools)} tools:")
    for t in tools:
        print(f"  - {t['name']} ({t['category']})")
    print()
    
    # Test cases
    tests = [
        ("wikipedia_search", {"query": "machine learning", "limit": 3}),
        ("wikipedia_summary", {"title": "Python (programming language)"}),
        ("wikidata_search", {"query": "California", "limit": 3}),
        ("worldbank_search_indicators", {"query": "GDP", "limit": 3}),
        ("coingecko_search", {"query": "bitcoin"}),
        ("coingecko_price", {"coin_id": "bitcoin", "vs": "usd"}),
        ("gdelt_news_search", {"query": "artificial intelligence", "max_results": 3}),
        ("arxiv_search", {"query": "large language models", "max_results": 3}),
        ("datagov_search", {"query": "climate", "rows": 3}),
    ]
    
    passed = 0
    failed = 0
    
    for tool_name, kwargs in tests:
        print(f"{'─' * 70}")
        print(f"Testing: {tool_name}")
        print(f"Input: {kwargs}")
        print(f"{'─' * 70}")
        
        result = call_tool(tool_name, **kwargs)
        
        if result.success:
            print(f"  ✓ Success ({result.response_time_ms}ms)")
            # Show preview of data
            if isinstance(result.data, list):
                print(f"  → {len(result.data)} results")
                for item in result.data[:2]:
                    if isinstance(item, tuple):
                        print(f"    - {item[0][:50] if item else 'N/A'}...")
                    else:
                        print(f"    - {str(item)[:50]}...")
            elif isinstance(result.data, dict):
                print(f"  → {result.data}")
            elif isinstance(result.data, tuple):
                print(f"  → {result.data[0][:50] if result.data[0] else 'N/A'}...")
            passed += 1
        else:
            print(f"  ✗ Error: {result.error}")
            failed += 1
        
        print()
    
    # Show database audit log
    print("=" * 70)
    print("DATABASE AUDIT LOG")
    print("=" * 70)
    
    try:
        db = Database.from_config(config_path)
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT tool_name, LEFT(input_params, 50) as input, 
                       status, response_time_ms, output_summary
                FROM tool_calls 
                ORDER BY id DESC 
                LIMIT 10
            """)
            rows = cursor.fetchall()
        
        print(f"\nLast {len(rows)} tool calls:")
        for r in rows:
            status = "✓" if r["status"] == "success" else "✗"
            print(f"  {status} {r['tool_name']}: {r['input'][:40]}... → {r['output_summary'] or r['status']} ({r['response_time_ms']}ms)")
        
        db.close()
    except Exception as e:
        print(f"  Could not read audit log: {e}")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{passed + failed}")
    
    if failed > 0:
        print(f"Failed: {failed}")
        return 1
    
    print("\n✓ All tool tests passed!")
    return 0


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "infra/config.yaml"
    raise SystemExit(test_tools(config))
