#!/usr/bin/env python3
"""
Update config.yaml in place with models from `ollama list`.

- Reads `ollama list`
- Extracts model names (NAME column)
- Writes them to endpoints.ollama.models as a YAML list

Usage:
  python3 update_ollama_models.py config.yaml
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from ruamel.yaml import YAML


def get_ollama_models() -> list[str]:
    # `ollama list` prints a table: NAME ID SIZE MODIFIED
    p = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True)
    lines = [ln.rstrip("\n") for ln in p.stdout.splitlines() if ln.strip()]

    # Find header row and parse everything after it.
    # We assume NAME is the first column, separated by 2+ spaces.
    header_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*NAME\s+ID\s+SIZE\s+MODIFIED\s*$", ln):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find expected header row in `ollama list` output.")

    models: list[str] = []
    for ln in lines[header_idx + 1 :]:
        # Split by 2+ spaces to avoid issues with single spaces in dates/etc.
        parts = re.split(r"\s{2,}", ln.strip())
        if not parts:
            continue
        name = parts[0].strip()
        if name and name != "NAME":
            models.append(name)

    if not models:
        raise RuntimeError("No models found from `ollama list`.")
    return models


def update_yaml_in_place(yaml_path: Path, models: list[str]) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True

    data = yaml.load(yaml_path.read_text())

    # Ensure structure exists
    data.setdefault("endpoints", {})
    data["endpoints"].setdefault("ollama", {})

    # Update models
    data["endpoints"]["ollama"]["models"] = models

    # Write back in place
    with yaml_path.open("w") as f:
        yaml.dump(data, f)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 update_ollama_models.py /path/to/config.yaml", file=sys.stderr)
        return 2

    yaml_path = Path(sys.argv[1]).expanduser().resolve()
    if not yaml_path.exists():
        print(f"Error: file not found: {yaml_path}", file=sys.stderr)
        return 2

    models = get_ollama_models()
    update_yaml_in_place(yaml_path, models)

    print(f"Updated {yaml_path} with {len(models)} models:")
    for m in models:
        print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

