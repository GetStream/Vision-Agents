#!/bin/bash

echo "🏏 Starting Cricket DRS AI System"
echo "=================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Start token server
echo "Starting token server on port 8001..."
python token_server.py &
TOKEN_PID=$!
sleep 2

# Start DRS agent (--no-demo prevents Stream's pronto UI from opening)
echo "Starting DRS AI agent..."
python cricket_umpire.py run --call-id cricket-umpire-agent &
AGENT_PID=$!
sleep 3

echo ""
echo "✅ Token server running (PID: $TOKEN_PID)"
echo "✅ DRS agent running (PID: $AGENT_PID)"
echo ""
echo "Open http://localhost:5173 in your browser"
echo ""

trap "kill $TOKEN_PID $AGENT_PID 2>/dev/null" EXIT

cd frontend
npm run dev