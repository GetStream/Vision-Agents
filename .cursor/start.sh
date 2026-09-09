#!/usr/bin/env bash
# Per-boot services for Cursor cloud agents. Idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/usr/local/go/bin:$HOME/go/bin:$HOME/.local/bin:$PATH"
export GOPRIVATE="${GOPRIVATE:-github.com/GetStream/*}"
export GONOSUMDB="${GONOSUMDB:-github.com/GetStream/*}"
export GOPROXY="${GOPROXY:-https://proxy.golang.org,direct}"
if [ -n "${GH_TOKEN:-}" ]; then
  git config --global "url.https://x-access-token:${GH_TOKEN}@github.com/GetStream/getstream-go-webrtc.insteadOf" \
    "https://github.com/GetStream/getstream-go-webrtc"
fi

sudo service postgresql start
sudo service redis-server start

PG_USER="$(whoami)"
PG_PASSWORD="${ROUTER_POSTGRES_PASSWORD:-voicebench}"
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$PG_USER'" | grep -q 1; then
  sudo -u postgres createuser --superuser "$PG_USER"
fi
# The router dials over TCP with a password: pgx reads host=/var/run/postgresql as the
# socket itself rather than the directory holding it, so the peer-auth DSN psql accepts
# fails the router's own migration with "permission denied".
sudo -u postgres psql -tAc "ALTER ROLE \"$PG_USER\" WITH PASSWORD '$PG_PASSWORD'" >/dev/null
if ! psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='model_router'" | grep -q 1; then
  createdb model_router
fi

export ROUTER_POSTGRES_DSN="${ROUTER_POSTGRES_DSN:-postgres://$PG_USER:$PG_PASSWORD@127.0.0.1:5432/model_router?sslmode=disable}"
export ROUTER_REDIS_ADDR="${ROUTER_REDIS_ADDR:-localhost:6379}"
export STREAM_ACCELERATION_CUSTOMER_ID="${STREAM_ACCELERATION_CUSTOMER_ID:-voicebench}"

python3 - "$ROOT" <<'PY'
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
keys: list[str] = []
example = root / ".env.example"
if example.is_file():
    for line in example.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.append(line.split("=", 1)[0])
for extra in (
    "STREAM_ACCELERATION_CUSTOMER_ID",
    "ROUTER_POSTGRES_DSN",
    "ROUTER_REDIS_ADDR",
    "GOPRIVATE",
    "GONOSUMDB",
):
    if extra not in keys:
        keys.append(extra)

google = os.environ.get("GOOGLE_API_KEY", "")
if google and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = google

skip = {"GH_TOKEN", "GITHUB_TOKEN", "GITHUB_PAT"}
lines: list[str] = []
for key in keys:
    if key in skip:
        continue
    value = os.environ.get(key)
    if not value:
        continue
    if value.startswith("your_") or value.endswith("_here"):
        continue
    lines.append(f"{key}={value}")

path = root / ".env"
path.write_text("\n".join(lines) + "\n")
path.chmod(stat.S_IRUSR | stat.S_IWUSR)
print(f"wrote {path} ({len(lines)} keys)", file=sys.stderr)
PY

goose_bin="$(command -v goose || true)"
if [ -z "$goose_bin" ] && [ -x "$HOME/go/bin/goose" ]; then
  goose_bin="$HOME/go/bin/goose"
fi
if [ -n "$goose_bin" ] && [ -d "$ROOT/acceleration/migrations" ]; then
  "$goose_bin" -dir "$ROOT/acceleration/migrations" postgres "$ROUTER_POSTGRES_DSN" up
fi
