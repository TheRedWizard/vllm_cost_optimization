#!/bin/bash
#
# Connect to the vLLM MySQL database
# Starts the container if not already running
#
# Usage: ./mysql.sh [query]
#   ./mysql.sh              # Interactive shell
#   ./mysql.sh "SELECT 1"   # Run a query
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Credentials (match docker-compose.yaml defaults)
MYSQL_USER="${MYSQL_USER:-vllm_user}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-vllm_pass}"
MYSQL_DATABASE="${MYSQL_DATABASE:-vllm}"

# Check if container exists and is running
if ! docker ps --format '{{.Names}}' | grep -q '^vllm_mysql$'; then
    echo "MySQL container not running. Starting..."
    ./setup_mysql.sh
    echo ""
fi

# Connect
if [ -n "$1" ]; then
    # Run query passed as argument
    docker exec vllm_mysql mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE" -e "$1"
else
    # Interactive shell
    echo "Connecting to MySQL ($MYSQL_DATABASE)..."
    docker exec -it vllm_mysql mysql -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DATABASE"
fi
