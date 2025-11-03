#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV=".venv"
if [ ! -d "$VENV" ]; then
  echo "[+] Creating venv"
  python -m venv "$VENV"
fi

# Activate venv across platforms (bin on Unix, Scripts on Windows/Git Bash)
if [ -f "$VENV/bin/activate" ]; then
  # Unix-like
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
elif [ -f "$VENV/Scripts/activate" ]; then
  # Windows venv under Git Bash
  # shellcheck disable=SC1090
  source "$VENV/Scripts/activate"
else
  echo "[-] Could not find venv activation script."
  exit 1
fi
# Use python -m pip to avoid Windows/Git Bash self-upgrade issues
python -m pip install -U pip >/dev/null
python -m pip install -r requirements.txt

# Load env file if present
if [ -f "env.sh" ]; then
  set -a
  source ./env.sh
  set +a
elif [ -f ".env" ]; then
  # python-dotenv will load this automatically, but we also export for clarity
  export $(grep -v '^#' .env | xargs -d '\n' -I{} bash -c 'k="${{1%%=*}}"; v="${{1#*=}}"; echo "$k=$v"' -- {})
fi

exec python live_rsi_bot.py "$@"
