#!/usr/bin/env bash
# Start step. Reports and photos live in $PETMATCH_DATA_DIR (default ./var). On a paid Render
# plan that is the persistent disk; on the free plan it is wiped on every restart, and the
# demo starts again from seed/demo.db.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${PETMATCH_DATA_DIR:-var}"
mkdir -p "$DATA_DIR/uploads"
export PETMATCH_DB="${PETMATCH_DB:-$DATA_DIR/petmatch.db}"
export PETMATCH_UPLOADS="${PETMATCH_UPLOADS:-$DATA_DIR/uploads}"

# First start of a demo: copy the prebuilt demo database into place (never overwrites).
if [ "${PETMATCH_DEMO:-0}" = "1" ] && [ ! -f "$PETMATCH_DB" ] && [ -f seed/demo.db ]; then
  cp seed/demo.db "$PETMATCH_DB"
  echo "Seeded $PETMATCH_DB with the demo reports."
fi

# One worker: SQLite has one writer, and one DINOv2 copy (about 350 MB) fits in 512 MB.
# --proxy-headers: Render's proxy passes the visitor's IP, which the rate limit needs.
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 \
  --proxy-headers --forwarded-allow-ips='*'
