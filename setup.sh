#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "🏎️  F1 Race Predictor — Setup"
echo "================================"

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 is required. Install from https://python.org"
  exit 1
fi

echo "✔  Python $(python3 --version)"

# Create virtualenv
if [ ! -d ".venv" ]; then
  echo "→  Creating virtual environment…"
  python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
echo "→  Installing Python dependencies…"
pip install --quiet --upgrade pip
pip install --quiet -r backend/requirements.txt

echo ""
echo "✅ Setup complete!"
echo "   Run: ./run.sh"
