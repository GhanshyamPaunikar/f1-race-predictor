#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "→  Running setup first…"
  bash setup.sh
fi

source .venv/bin/activate

echo ""
echo "🏎️  Starting F1 Race Predictor…"
echo "   Open: http://localhost:8001"
echo "   Press Ctrl+C to stop"
echo ""

cd backend
exec python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
