#!/bin/bash
#
# Full EC2 Setup Script for vLLM Cost Optimization
#
# This script will:
# 1. Install system dependencies (Docker, Python, Ollama)
# 2. Setup Python virtual environment
# 3. Start MySQL via Docker
# 4. Start Ollama and pull models
# 5. Run all tests
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/.../setup_ec2.sh | bash
#   # OR
#   chmod +x setup_ec2.sh && ./setup_ec2.sh
#
# Tested on: Amazon Linux 2023, Ubuntu 22.04
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo "═══════════════════════════════════════════════════════════════════"
echo "  vLLM Cost Optimization - EC2 Setup"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    error "Cannot detect OS"
fi

log "Detected OS: $OS"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Install System Dependencies
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 1: Installing system dependencies"
echo "─────────────────────────────────────────────────────────────────────"

install_docker() {
    if command -v docker &> /dev/null; then
        log "Docker already installed: $(docker --version)"
        return
    fi
    
    warn "Installing Docker..."
    
    if [ "$OS" = "amzn" ]; then
        # Amazon Linux
        sudo yum update -y
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
    elif [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/$OS/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        sudo usermod -aG docker $USER
    else
        error "Unsupported OS for Docker install: $OS"
    fi
    
    log "Docker installed"
}

install_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        log "Python already installed: $PYTHON_VERSION"
        return
    fi
    
    warn "Installing Python..."
    
    if [ "$OS" = "amzn" ]; then
        sudo yum install -y python3 python3-pip python3-venv
    elif [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    fi
    
    log "Python installed"
}

install_ollama() {
    if command -v ollama &> /dev/null; then
        log "Ollama already installed: $(ollama --version)"
        return
    fi
    
    warn "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    log "Ollama installed"
}

install_docker
install_python
install_ollama

# ─────────────────────────────────────────────────────────────────────────────
# 2. Setup Project
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 2: Setting up project"
echo "─────────────────────────────────────────────────────────────────────"

# Find project root (where this script lives is infra/, go up one level)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
log "Project root: $PROJECT_ROOT"

# Create virtual environment if not exists
if [ ! -d "env/venv" ]; then
    warn "Creating Python virtual environment..."
    mkdir -p env
    python3 -m venv env/venv
    log "Virtual environment created"
else
    log "Virtual environment exists"
fi

# Activate venv
source env/venv/bin/activate
log "Virtual environment activated"

# Install Python dependencies
warn "Installing Python packages..."
pip install --upgrade pip -q
pip install requests ruamel.yaml mysql-connector-python feedparser -q
log "Python packages installed"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Start MySQL
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 3: Starting MySQL"
echo "─────────────────────────────────────────────────────────────────────"

# Need to use sudo for docker if user not in docker group yet
if groups $USER | grep -q docker; then
    DOCKER_CMD="docker"
else
    warn "User not in docker group yet. Using sudo for docker commands."
    warn "Log out and back in after this script to use docker without sudo."
    DOCKER_CMD="sudo docker"
fi

# Check if MySQL container exists
if $DOCKER_CMD ps -a --format '{{.Names}}' | grep -q '^vllm_mysql$'; then
    if $DOCKER_CMD ps --format '{{.Names}}' | grep -q '^vllm_mysql$'; then
        log "MySQL container already running"
    else
        warn "Starting existing MySQL container..."
        $DOCKER_CMD start vllm_mysql
        sleep 5
        log "MySQL container started"
    fi
else
    warn "Creating and starting MySQL container..."
    cd "$PROJECT_ROOT/infra"
    
    # Use docker compose (with sudo if needed)
    if groups $USER | grep -q docker; then
        docker compose up -d
    else
        sudo docker compose up -d
    fi
    
    # Wait for MySQL to be ready
    echo -n "Waiting for MySQL to be ready"
    for i in {1..30}; do
        if $DOCKER_CMD exec vllm_mysql mysqladmin ping -h localhost -u root -pvllm_root_pass --silent 2>/dev/null; then
            echo ""
            log "MySQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            echo ""
            error "MySQL failed to start within 30 seconds"
        fi
        echo -n "."
        sleep 1
    done
    
    cd "$PROJECT_ROOT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Start Ollama and Pull Models
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 4: Starting Ollama and pulling models"
echo "─────────────────────────────────────────────────────────────────────"

# Start Ollama service if not running
if ! pgrep -x "ollama" > /dev/null; then
    warn "Starting Ollama service..."
    
    # Try systemd first
    if systemctl is-active --quiet ollama 2>/dev/null; then
        log "Ollama service already running via systemd"
    elif command -v systemctl &> /dev/null && systemctl list-unit-files | grep -q ollama; then
        sudo systemctl start ollama
        sleep 2
        log "Ollama started via systemd"
    else
        # Start manually in background
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3
        log "Ollama started manually"
    fi
else
    log "Ollama already running"
fi

# Wait for Ollama to be ready
echo -n "Waiting for Ollama API"
for i in {1..30}; do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo ""
        log "Ollama API is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo ""
        error "Ollama API failed to respond"
    fi
    echo -n "."
    sleep 1
done

# Pull required models
MODELS=(
    "qwen3:8b"
    "qwen2.5-coder:7b"
    "nomic-embed-text:latest"
)

for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        log "Model already pulled: $model"
    else
        warn "Pulling model: $model (this may take a while)..."
        ollama pull "$model"
        log "Model pulled: $model"
    fi
done

# Update config with all available models
warn "Updating config with available models..."
python3 infra/update_ollama_models.py infra/config.yaml
log "Config updated"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Run Tests
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Step 5: Running tests"
echo "─────────────────────────────────────────────────────────────────────"

# Test models
warn "Testing chat and embedding models..."
python3 src/test_models.py

echo ""

# Test tools
warn "Testing external data source tools..."
python3 src/test_tools.py

echo ""

# Test reproducibility (quick version)
warn "Testing reproducibility..."
python3 -c "
from src.ollama_client import OllamaClient
client = OllamaClient.from_config('infra/config.yaml')
r1 = client.chat('qwen3:8b', [{'role': 'user', 'content': 'Say hello'}], temperature=0, seed=42)
r2 = client.chat('qwen3:8b', [{'role': 'user', 'content': 'Say hello'}], temperature=0, seed=42)
if r1.content == r2.content:
    print('✓ Reproducibility test passed')
else:
    print('✗ Reproducibility test failed')
    exit(1)
"

# ─────────────────────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "To use:"
echo "  cd $PROJECT_ROOT"
echo "  source env/venv/bin/activate"
echo ""
echo "Quick test:"
echo "  python3 -c \"from src import call_model; print(call_model('qwen3:8b', 'Hello!'))\""
echo ""
echo "MySQL CLI:"
echo "  ./infra/mysql.sh"
echo ""
echo "Stop services:"
echo "  cd infra && docker compose down  # MySQL"
echo "  pkill ollama                      # Ollama"
echo ""

# Note about docker group
if ! groups $USER | grep -q docker; then
    echo "─────────────────────────────────────────────────────────────────────"
    echo "NOTE: Log out and back in to use docker without sudo"
    echo "─────────────────────────────────────────────────────────────────────"
fi
