#!/usr/bin/env bash
# Start the PathSnap API on Laptop 1.
# Usage (from the project folder):  bash scripts/start_api.sh
# Settings come from .env (copy .env.example to .env) or from environment variables:
#   PORT=8000  PATHSNAP_PARTITION=student_test  PATHSNAP_CORS_ORIGINS=*  PATHSNAP_EVALUATE=1
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f venv/bin/activate ]; then source venv/bin/activate; fi
PORT="${PORT:-8000}"
export PATHSNAP_PARTITION="${PATHSNAP_PARTITION:-student_test}"
echo "Serving partition: ${PATHSNAP_PARTITION}  (data/raw/${PATHSNAP_PARTITION})"
echo "This machine's addresses (give one of these to Laptop 2):"
hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | sed "s|^|   http://|; s|$|:${PORT}|" || true
echo "Request log: logs/api.log   (Ctrl+C to stop)"
exec uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port "${PORT}"
