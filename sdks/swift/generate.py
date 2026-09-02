#!/usr/bin/env python
"""Regenerate the Swift client from the acceleration OpenAPI spec.

The spec at ``acceleration/api/openapi.yaml`` is the source of truth. The output is committed
so building the package needs no code generation and no build-tool plugin to be trusted in
Xcode, which matches what the Python and Go clients already do.

Only the operations a phone is allowed to call are generated. The filter is the point: an
endpoint marked ``x-server-side-only`` in the spec configures an agent or listens for
dispatched calls, and a client SDK that had a method for it would only be offering callers a
403. ``--check`` verifies the filter still agrees with the spec without regenerating.

WebSockets are not generated. OpenAPI cannot describe a socket past the upgrade, so the
session events socket is hand-written in ``SessionSocket.swift``.

Usage:
    uv run sdks/swift/generate.py
    uv run sdks/swift/generate.py --check
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SWIFT = Path(__file__).parent
ROOT = SWIFT.parents[1]
SPEC = ROOT / "acceleration" / "api" / "openapi.yaml"
TARGET = SWIFT / "core" / "Sources" / "VisionAgentsCore"
GENERATED = TARGET / "Generated"

# What a client is allowed to reach. Sessions and the two token endpoints are the whole of
# holding a conversation; the configs and calls reads are what an app shows about one.
OPERATIONS = [
    "closeSession",
    "createCallToken",
    "createChatToken",
    "createSession",
    "getAgentConfig",
    "getCall",
    "getCallTranscript",
    "getSession",
    "interruptSession",
    "listAgentConfigs",
    "listCalls",
    "listSessions",
    "respondSession",
    "saySession",
    "setSessionInstructions",
]

CONFIG = {
    "generate": ["types", "client"],
    "accessModifier": "internal",
    "namingStrategy": "idiomatic",
    "filter": {"operations": OPERATIONS},
}


def audit(spec: dict) -> list[str]:
    """Report the named operations that the spec says are server-side only, or unknown."""
    found: dict[str, dict] = {}
    for operations in spec["paths"].values():
        for operation in operations.values():
            if isinstance(operation, dict) and "operationId" in operation:
                found[operation["operationId"]] = operation

    complaints = []
    for name in OPERATIONS:
        operation = found.get(name)
        if operation is None:
            complaints.append(f"{name} is not in the spec")
        elif operation.get("x-server-side-only"):
            complaints.append(
                f"{name} is server-side only and cannot be in a client SDK"
            )
    return complaints


def main(argv: list[str]) -> int:
    if not SPEC.exists():
        print(f"no spec at {SPEC}", file=sys.stderr)
        return 1

    spec = yaml.safe_load(SPEC.read_text())
    complaints = audit(spec)
    if complaints:
        for complaint in complaints:
            print(complaint, file=sys.stderr)
        return 1

    if "--check" in argv:
        print(f"{len(OPERATIONS)} client operations, none server-side only")
        return 0

    generator = ROOT / ".codegen" / "swift" / "swift-openapi-generator"
    if not generator.exists():
        print(f"no generator at {generator}; see sdks/swift/README.md", file=sys.stderr)
        return 1

    work = SWIFT / ".codegen"
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)

    config = work / "openapi-generator-config.yaml"
    config.write_text(yaml.safe_dump(CONFIG, sort_keys=False))

    result = subprocess.run(
        [
            str(generator),
            "generate",
            str(SPEC),
            "--config",
            str(config),
            "--output-directory",
            str(work / "out"),
        ]
    )
    if result.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        return result.returncode

    shutil.rmtree(GENERATED, ignore_errors=True)
    GENERATED.mkdir(parents=True)
    for generated in sorted((work / "out").glob("*.swift")):
        shutil.move(str(generated), str(GENERATED / generated.name))
    shutil.rmtree(work, ignore_errors=True)

    print(f"regenerated {GENERATED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
