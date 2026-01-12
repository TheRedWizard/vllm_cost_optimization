#!/bin/bash
#
# Setup MySQL Docker container for vLLM cost optimization
#
# Usage: ./setup_mysql.sh [--reset]
#   --reset: Remove existing container and data, start fresh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default credentials (can be overridden via environment)
export MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-vllm_root_pass}"
export MYSQL_DATABASE="${MYSQL_DATABASE:-vllm}"
export MYSQL_USER="${MYSQL_USER:-vllm_user}"
export MYSQL_PASSWORD="${MYSQL_PASSWORD:-vllm_pass}"
export MYSQL_PORT="${MYSQL_PORT:-3306}"

echo "═══════════════════════════════════════════════════════════"
echo "  vLLM MySQL Setup"
echo "═══════════════════════════════════════════════════════════"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Handle --reset flag
if [[ "$1" == "--reset" ]]; then
    echo "Resetting MySQL container and data..."
    docker compose down -v 2>/dev/null || true
    echo "Reset complete."
fi

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q '^vllm_mysql$'; then
    echo "MySQL container is already running."
    echo ""
    echo "Connection details:"
    echo "  Host: localhost"
    echo "  Port: $MYSQL_PORT"
    echo "  Database: $MYSQL_DATABASE"
    echo "  User: $MYSQL_USER"
    echo "  Password: $MYSQL_PASSWORD"
    exit 0
fi

# Start the container
echo "Starting MySQL container..."
docker compose up -d

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
for i in {1..30}; do
    if docker exec vllm_mysql mysqladmin ping -h localhost -u root -p"$MYSQL_ROOT_PASSWORD" --silent 2>/dev/null; then
        echo "MySQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: MySQL failed to start within 30 seconds"
        docker logs vllm_mysql
        exit 1
    fi
    sleep 1
    echo -n "."
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  MySQL is running!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Connection details:"
echo "  Host: localhost"
echo "  Port: $MYSQL_PORT"
echo "  Database: $MYSQL_DATABASE"
echo "  User: $MYSQL_USER"
echo "  Password: $MYSQL_PASSWORD"
echo ""
echo "To connect via CLI:"
echo "  docker exec -it vllm_mysql mysql -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE"
echo ""
echo "To stop:"
echo "  cd $SCRIPT_DIR && docker compose down"
echo ""
