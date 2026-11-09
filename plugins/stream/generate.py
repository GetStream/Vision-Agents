#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""Regenerate the HTTP client from the acceleration OpenAPI spec.

The spec at ``acceleration/api/openapi.yaml`` is the source of truth for both sides: Go
generates its server from it and this generates the Python client. The output is committed
so installing the plugin needs no code generation, and running this after changing the spec
is what keeps the two in step.

WebSockets are not generated. OpenAPI cannot describe a socket past the upgrade, so the
events and modality streams are hand-written in ``_ws.py``. SSE operations are also
excluded: the generated HTTP client buffers entire responses and cannot consume an
open-ended event stream. The dashboard consumes agent logs using EventSource. Operations
tagged ``Staff`` are Stream's own and are left out as well.

Usage:
    uv run plugins/stream/generate.py
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PLUGIN = Path(__file__).parent
SPEC = PLUGIN.parents[1] / "acceleration" / "api" / "openapi.yaml"
PACKAGE = PLUGIN / "vision_agents" / "plugins" / "stream"
GENERATED = PACKAGE / "_generated"
STAFF_TAG = "Staff"
# Templates that replace the generator's own: an enum read from an answer keeps a value this
# client does not list yet, rather than raising, so a router that adds one breaks no reader.
TEMPLATES = PLUGIN / "templates"

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

    # The generator treats text/event-stream as a finite string response, which
    # would buffer indefinitely. Keep streaming endpoints out of the REST client.
    # Operations tagged Staff (/v1/ops/*, /v1/manage/*) are Stream's own, reached
    # with a staff key and never by a customer, so they stay out too.
    document = yaml.safe_load(SPEC.read_text())
    for path, item in list(document["paths"].items()):
        for method, operation in list(item.items()):
            if not isinstance(operation, dict):
                continue
            if STAFF_TAG in operation.get("tags", []) or any(
                "text/event-stream" in response.get("content", {})
                for response in operation.get("responses", {}).values()
            ):
                del item[method]
        if not item:
            del document["paths"][path]
    keep_released_order(document["components"]["schemas"], released_orders())
    http_spec = work / "http-openapi.yaml"
    http_spec.write_text(yaml.safe_dump(document, sort_keys=False))

    config = work / "config.yaml"
    config.write_text(CONFIG)

    result = subprocess.run(
        [
            "uvx",
            "--from",
            "openapi-python-client==0.29.1",
            "openapi-python-client",
            "generate",
            "--path",
            str(http_spec),
            "--config",
            str(config),
            "--custom-template-path",
            str(TEMPLATES),
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


def released_orders() -> dict[str, list[str]]:
    """The wire names of each committed model's fields, in the order its attrs class takes them.

    from_dict pops every field in that order, under its wire name.
    """
    orders: dict[str, list[str]] = {}
    for path in (GENERATED / "models").glob("*.py"):
        source = path.read_text()
        named = re.search(r"^class (\w+)", source, re.MULTILINE)
        if named:
            orders[named.group(1)] = re.findall(r'd\.pop\("([^"]+)"', source)
    return orders


def keep_released_order(schemas: dict, orders: dict[str, list[str]]) -> None:
    """Puts each schema's properties in the order its committed model takes them, new ones last.

    The spec lists properties alphabetically and an attrs class takes its fields positionally,
    required ones first, in the spec's order. A new property would otherwise shift those after
    it, and a positional call written against an earlier release would pass its arguments to
    the wrong fields.
    """
    for name, schema in schemas.items():
        released = orders.get(name)
        properties = schema.get("properties")
        if not released or not isinstance(properties, dict):
            continue
        place = {wire: index for index, wire in enumerate(released)}
        schema["properties"] = dict(
            sorted(properties.items(), key=lambda item: place.get(item[0], len(place)))
        )


if __name__ == "__main__":
    raise SystemExit(main())
