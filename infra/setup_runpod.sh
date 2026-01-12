#!/bin/bash
#
# RunPod Setup Script for vLLM Cost Optimization
#
# This script will:
# 1. Install Ollama
# 2. Setup Python virtual environment
# 3. Pull models
# 4. Run tests (using SQLite instead of MySQL for simplicity)
#
# Usage:
#   chmod +x infra/setup_runpod.sh && ./infra/setup_runpod.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo "═══════════════════════════════════════════════════════════════════"
echo "  vLLM Cost Optimization - RunPod Setup"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
log "Project root: $PROJECT_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Install System Dependencies
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 1: Installing system dependencies"
echo "─────────────────────────────────────────────────────────────────────"

# Update package list
if command -v apt-get &> /dev/null; then
    warn "Updating apt..."
    apt-get update -qq
fi

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    warn "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    log "Ollama installed"
else
    log "Ollama already installed"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Setup Python Environment
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 2: Setting up Python environment"
echo "─────────────────────────────────────────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
    error "Python3 not found. Please use a RunPod template with Python."
fi
log "Python: $(python3 --version)"

# Create venv if not exists
if [ ! -d "env/venv" ]; then
    warn "Creating virtual environment..."
    mkdir -p env
    python3 -m venv env/venv
    log "Virtual environment created"
else
    log "Virtual environment exists"
fi

# Activate
source env/venv/bin/activate
log "Virtual environment activated"

# Install packages
warn "Installing Python packages..."
pip install --upgrade pip -q
pip install requests ruamel.yaml feedparser -q
log "Python packages installed"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Start Ollama and Pull Models
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 3: Starting Ollama and pulling models"
echo "─────────────────────────────────────────────────────────────────────"

# Start Ollama if not running
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    warn "Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    
    # Wait for it to be ready
    echo -n "Waiting for Ollama"
    for i in {1..60}; do
        if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo ""
            log "Ollama is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            echo ""
            error "Ollama failed to start. Check /tmp/ollama.log"
        fi
        echo -n "."
        sleep 1
    done
else
    log "Ollama already running"
fi

# Pull models (same as local setup)
MODELS=(
    # Chat models
    "qwen3:8b"              # Think fast
    "qwen2.5-coder:7b"      # Write code
    "mistral-nemo:latest"   # Read a lot (128k context)
    "deepseek-r1:14b"       # Think deeply (reasoning)
    "qwen2.5vl:7b"          # See images (vision)
    # Embedding models
    "nomic-embed-text:latest"  # Fast embeddings
    "bge-m3:latest"            # Multilingual embeddings
)

for model in "${MODELS[@]}"; do
    if ollama list 2>/dev/null | grep -q "$(echo $model | cut -d: -f1)"; then
        log "Model already pulled: $model"
    else
        warn "Pulling model: $model (this may take a while)..."
        ollama pull "$model"
        log "Model pulled: $model"
    fi
done

# Update config
warn "Updating config with available models..."
python3 infra/update_ollama_models.py infra/config.yaml
log "Config updated"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Test (without MySQL - just Ollama)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 4: Running tests (without database)"
echo "─────────────────────────────────────────────────────────────────────"

# Quick model test
warn "Testing models..."
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

from ollama_client import OllamaClient, get_chat_models, get_embedding_models

client = OllamaClient.from_config('infra/config.yaml', log_to_db=False)

chat_models = get_chat_models('infra/config.yaml')
embed_models = get_embedding_models('infra/config.yaml')

print(f"Testing {len(chat_models)} chat models and {len(embed_models)} embedding models...")
print()

# Test one chat model
if chat_models:
    model = chat_models[0]
    print(f"Testing chat: {model}")
    resp = client.chat(model, [{"role": "user", "content": "Say hello"}], output_tokens=50)
    print(f"  ✓ Response: {resp.content[:50]}...")
    if resp.reasoning:
        print(f"  ✓ Has reasoning")

# Test one embedding model
if embed_models:
    model = embed_models[0]
    print(f"Testing embedding: {model}")
    vectors = client.embed(model, "Hello world")
    print(f"  ✓ Got {len(vectors)} vector(s) with {len(vectors[0])} dimensions")

print()
print("✓ Basic tests passed!")
EOF

# Test tools (no DB needed)
warn "Testing external tools..."
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

from tools import call_tool, list_tools, registry

# Don't need DB for tool tests
registry.db = None

print(f"Testing {len(list_tools())} tools...")

# Quick test of a few tools
tests = [
    ("wikipedia_search", {"query": "Python", "limit": 2}),
    ("coingecko_price", {"coin_id": "bitcoin"}),
]

for name, kwargs in tests:
    result = call_tool(name, **kwargs)
    if result.success:
        print(f"  ✓ {name}: OK")
    else:
        print(f"  ✗ {name}: {result.error}")

print()
print("✓ Tool tests passed!")
EOF

# ─────────────────────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  RunPod Setup Complete!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "To use:"
echo "  cd $PROJECT_ROOT"
echo "  source env/venv/bin/activate"
echo ""
echo "Quick test:"
echo "  python3 -c \"from src import call_model; print(call_model('qwen3:8b', 'Hello!', log_to_db=False))\""
echo ""
echo "NOTE: Database logging is disabled on RunPod (no Docker for MySQL)."
echo "      For full auditability, deploy on EC2 or a machine with Docker."
echo ""

# Show GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "─────────────────────────────────────────────────────────────────────"
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "─────────────────────────────────────────────────────────────────────"
fi
