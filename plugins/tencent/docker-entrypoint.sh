#!/bin/sh
set -e

uv sync

SITE_PACKAGES=$(uv run python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
cp /opt/liteav/ext/_liteav*.so "$SITE_PACKAGES/"

exec "$@"
