#!/usr/bin/env bash
# Quick launcher for Multiverse™ 3D Spatial Platform & RAG Chatbot Studio
set -e
cd "$(dirname "$0")"

PORT="${1:-8080}"

echo "============================================================================"
echo " Starting Multiverse™ 3D Spatial Platform & RAG Studio on port $PORT"
echo " Web 3D Interface & Chatbot: http://localhost:$PORT"
echo "============================================================================"

if [ -f ".venv/bin/python3" ]; then
    exec .venv/bin/python3 server.py "$PORT"
else
    exec python3 server.py "$PORT"
fi
