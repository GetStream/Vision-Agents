#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""Regenerate the Swift client from the acceleration OpenAPI spec.

The spec at ``acceleration/api/openapi.yaml`` is the source of truth. The client is written by
the generator Stream Chat and Stream Video use, from the ``chat`` repository checked out beside
this one (``tools/openapi``), so the models and ``DefaultAPI`` lean on the same StreamCore
runtime as theirs. The output is committed, so building the package needs no code generation
and no Go.

Only the operations a phone is allowed to call are generated. The filter is the point: an
endpoint the spec does not mark ``x-client-accessible`` is server-side only, and a client
SDK that had a method for one would only be offering callers a 403. ``--check`` verifies the
filter still agrees with the spec without regenerating.

The generator is handed a copy of the spec holding those operations, their successes and the
schemas they reach, and nothing else: ``FailureMiddleware`` throws every other status before the
client reads it. The copy is also put in terms the generator reads. It reads only string enums,
so an integer enum is a plain integer, and it drops a field that is a string-or-object union,
so such a field is sent as its string shorthand. It ignores ``nullable``, so a required field
that may be null is not required, or a null would fail to decode. A model whose name a
hand-written type already has is renamed, as ``RENAMED`` says.

WebSockets are not generated. OpenAPI cannot describe a socket past the upgrade, so the
session events socket is hand-written in ``SessionSocket.swift``.

Usage:
    uv run sdks/swift/generate.py
    uv run sdks/swift/generate.py --check
"""

import json
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
GENERATOR = ROOT.parent / "chat" / "tools" / "openapi"

# What a client is allowed to reach: opening a conversation, finding and ending one, a guest
# to do it as, looking something up, and the user's own scheduled tasks. Everything else about
# an agent — what it is configured as, what it has said before, a token to join a call with, and the routed modalities on their own — is
# the app's own backend to ask for and to hand down. Holding the conversation is driven over
# the session socket, which is hand-written rather than generated.
OPERATIONS = [
    "createGuestUser",
    "createResponse",
    "createSession",
    "deleteScheduledTask",
    "deleteSession",
    "forkSession",
    "getScheduledTask",
    "getSession",
    "listResponseItems",
    "listResponses",
    "listScheduledTaskRuns",
    "listScheduledTasks",
    "pauseScheduledTask",
    "querySessions",
    "resumeScheduledTask",
    "rewindSession",
    "search",
    "startSessionVoice",
    "stopSession",
    "stopSessionVoice",
    "updateScheduledTask",
    "updateSession",
]

# The generator puts every model at the top level of the module, beside the hand-written types
# that wrap them, so a model sharing a name with one is renamed.
RENAMED = {
    "Greeting": "GreetingPayload",
    "ImageSource": "ImageSourcePayload",
    "ModelOverwrites": "ModelOverwritesPayload",
    "ScheduledTask": "ScheduledTaskPayload",
    "ScheduledTaskRun": "ScheduledTaskRunPayload",
    "SearchAnswer": "SearchAnswerPayload",
    "SearchOptions": "SearchOptionsPayload",
    "Session": "SessionPayload",
    "SessionQuery": "SessionQueryPayload",
}

# What Stream Chat's SDK asks of the generator too: internal, immutable models, enums that
# keep a value the router added later, and one shared set of coding keys.
OPTIONS = [
    "access_modifier=internal",
    "immutable_models=true",
    "raw_representable_over_enum=true",
    "string_coding_keys=true",
]


def audit(spec: dict) -> list[str]:
    """Report the named operations the spec does not open to a client, or does not have."""
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
        elif not operation.get("x-client-accessible"):
            complaints.append(
                f"{name} is server-side only and cannot be in a client SDK"
            )
    return complaints


def references(node: object) -> set[str]:
    """Every ``$ref`` under ``node``."""
    if isinstance(node, list):
        return set().union(set(), *map(references, node))
    if not isinstance(node, dict):
        return set()
    found = {node["$ref"]} if isinstance(node.get("$ref"), str) else set()
    return found.union(*map(references, node.values()))


def client(spec: dict) -> dict:
    """The spec as the generator should see it: the client operations with their 2xx
    responses, and the components they reach."""
    paths: dict[str, dict] = {}
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if (
                isinstance(operation, dict)
                and operation.get("operationId") in OPERATIONS
            ):
                operation["responses"] = {
                    status: response
                    for status, response in operation["responses"].items()
                    if status.startswith("2")
                }
                paths.setdefault(path, {})[method] = operation

    reached: set[str] = set()
    pending = references(paths)
    while pending:
        ref = pending.pop()
        reached.add(ref)
        _, _, kind, name = ref.split("/")
        pending |= references(spec["components"][kind][name]) - reached

    components: dict[str, dict] = {}
    for ref in sorted(reached):
        _, _, kind, name = ref.split("/")
        components.setdefault(kind, {})[name] = spec["components"][kind][name]
    spec["paths"] = paths
    spec["components"] = components
    return spec


def readable(node: object, schemas: dict) -> object:
    """``node`` in terms the generator reads: no integer enums, no required field that may be
    null, and a reference to a string-or-object union written out as a string where it is
    used."""
    if isinstance(node, list):
        return [readable(value, schemas) for value in node]
    if not isinstance(node, dict):
        return node
    union = schemas.get(node.get("$ref", "").removeprefix("#/components/schemas/"), {})
    if {"type": "string"} in union.get("oneOf", []):
        node = {key: value for key, value in node.items() if key != "$ref"}
        node["type"] = "string"
    node = {key: readable(value, schemas) for key, value in node.items()}
    if node.get("type") == "integer":
        node.pop("enum", None)
    # A parameter's or a body's required is a flag; only a schema's is a list of fields.
    if isinstance(node.get("required"), list):
        properties = node.get("properties", {})
        node["required"] = [
            name
            for name in node["required"]
            if not properties.get(name, {}).get("nullable")
        ]
    return node


def renamed(spec: dict) -> dict:
    """``spec`` with the models in ``RENAMED`` under their new names."""
    text = json.dumps(spec)
    for old, new in RENAMED.items():
        text = text.replace(
            f'"#/components/schemas/{old}"', f'"#/components/schemas/{new}"'
        )
    spec = json.loads(text)
    schemas = spec["components"]["schemas"]
    spec["components"]["schemas"] = {
        RENAMED.get(name, name): schema for name, schema in schemas.items()
    }
    return spec


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

    if not GENERATOR.exists():
        print(f"no generator at {GENERATOR}; see sdks/swift/README.md", file=sys.stderr)
        return 1

    work = SWIFT / ".codegen"
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)

    trimmed = work / "openapi.json"
    trimmed.write_text(
        json.dumps(renamed(client(readable(spec, spec["components"]["schemas"]))))
    )

    command = ["go", "run", ".", "generate-client", "--language", "swift"]
    for option in OPTIONS:
        command += ["--opt", option]
    command += ["--spec", str(trimmed), "--output", str(work / "out")]
    result = subprocess.run(command, cwd=GENERATOR)
    if result.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        return result.returncode

    # The endpoint factories are Stream Chat's own request path, written against its internals.
    # This SDK calls DefaultAPI.
    shutil.rmtree(GENERATED, ignore_errors=True)
    (GENERATED / "Models").mkdir(parents=True)
    shutil.move(
        str(work / "out" / "APIs" / "DefaultAPI.swift"),
        str(GENERATED / "DefaultAPI.swift"),
    )
    shutil.move(
        str(work / "out" / "StringCodingKey.swift"),
        str(GENERATED / "StringCodingKey.swift"),
    )
    for generated in sorted((work / "out" / "models").glob("*.swift")):
        shutil.move(str(generated), str(GENERATED / "Models" / generated.name))
    shutil.rmtree(work, ignore_errors=True)

    print(f"regenerated {GENERATED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
