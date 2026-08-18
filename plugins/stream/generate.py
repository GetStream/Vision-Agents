#!/usr/bin/env python
"""Regenerate the HTTP client from the acceleration OpenAPI spec.

The spec at ``acceleration/api/openapi.yaml`` is the source of truth for both sides: Go
generates its server from it and this generates the Python client. The output is committed
so installing the plugin needs no code generation, and running this after changing the spec
is what keeps the two in step.

WebSockets are not generated. OpenAPI cannot describe a socket past the upgrade, so the
events and modality streams are hand-written in ``_ws.py``.

Usage:
    uv run plugins/stream/generate.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN = Path(__file__).parent
SPEC = PLUGIN.parents[1] / "acceleration" / "api" / "openapi.yaml"
PACKAGE = PLUGIN / "vision_agents" / "plugins" / "stream"
GENERATED = PACKAGE / "_generated"

CONFIG = """
project_name_override: _generated
package_name_override: _generated
"""


def main() -> int:
    if not SPEC.exists():
        print(f"no spec at {SPEC}", file=sys.stderr)
        return 1

    work = PLUGIN / ".codegen"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()

    config = work / "config.yaml"
    config.write_text(CONFIG)

    result = subprocess.run(
        [
            "uvx",
            "--from",
            "openapi-python-client",
            "openapi-python-client",
            "generate",
            "--path",
            str(SPEC),
            "--config",
            str(config),
            "--output-path",
            str(work / "out"),
            "--overwrite",
            "--meta",
            "none",
        ],
        cwd=work,
    )
    if result.returncode != 0:
        return result.returncode

    shutil.rmtree(GENERATED, ignore_errors=True)
    shutil.move(str(work / "out"), str(GENERATED))
    shutil.rmtree(work, ignore_errors=True)

    print(f"regenerated {GENERATED.relative_to(PLUGIN.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
