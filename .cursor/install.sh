#!/usr/bin/env bash
# Build-time setup for Cursor cloud agents. Must be idempotent and terminate.
set -euo pipefail

GO_VERSION=1.25.7
PYTHON_VERSION=3.12.11

sudo apt-get update
# libopus/libopusfile + pkg-config are needed by the cgo Opus decoder in livekit media-sdk.
sudo apt-get install -y postgresql redis-server pkg-config libopus-dev libopusfile-dev

export PATH="/usr/local/go/bin:$HOME/go/bin:$HOME/.local/bin:$PATH"
if ! grep -q '/usr/local/go/bin' "$HOME/.bashrc"; then
  echo 'export PATH="/usr/local/go/bin:$HOME/go/bin:$HOME/.local/bin:$PATH"' >>"$HOME/.bashrc"
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv venv --python "$PYTHON_VERSION"
uv sync --all-extras --dev

if [ "$(go version 2>/dev/null | awk '{print $3}')" != "go${GO_VERSION}" ]; then
  curl -LsSf "https://go.dev/dl/go${GO_VERSION}.linux-$(dpkg --print-architecture).tar.gz" -o /tmp/go.tar.gz
  sudo rm -rf /usr/local/go
  sudo tar -C /usr/local -xzf /tmp/go.tar.gz
  rm /tmp/go.tar.gz
fi
go install github.com/pressly/goose/v3/cmd/goose@latest

# Give the agent's own user a Postgres superuser role so psql works without sudo.
sudo service postgresql start
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$(whoami)'" | grep -q 1; then
  sudo -u postgres createuser --superuser "$(whoami)"
fi
