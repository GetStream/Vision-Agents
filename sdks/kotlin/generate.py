#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Regenerate the Kotlin models from the acceleration OpenAPI spec.

The spec at ``acceleration/api/openapi.yaml`` is the source of truth. Only the models are
generated, into ``core/src/main/kotlin/.../generated``, and they are ``internal``: the request
layer is a screenful of Ktor and is hand-written, because a generated client would decide how a
token is attached and which engine to use, which is the part a phone needs to own. The output is
committed so building needs no generator.

Only what a phone may call is generated. The spec is pruned to the operations below and the
schemas they reach before the generator sees it, and an operation not marked
``x-client-accessible`` in the spec fails the run: a client SDK with a model for a server-side
operation is one step from a method that only ever answers 403. ``--check`` verifies the filter
still agrees with the spec without regenerating.

The generator is OpenAPI Generator in Docker, pinned below, so nothing is installed on the host.

Usage:
    uv run sdks/kotlin/generate.py
    uv run sdks/kotlin/generate.py --check
"""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

KOTLIN = Path(__file__).parent
ROOT = KOTLIN.parents[1]
SPEC = ROOT / "acceleration" / "api" / "openapi.yaml"
PACKAGE = "io.getstream.visionagents.core.generated"
GENERATED = KOTLIN / "core" / "src" / "main" / "kotlin" / Path(*PACKAGE.split("."))
IMAGE = "openapitools/openapi-generator-cli:v7.25.0"

# What a client is allowed to reach: opening a conversation, finding, reading and ending one,
# going back in it or branching off it, becoming a guest, and looking something up.
OPERATIONS = [
    "closeSession",
    "createGuestUser",
    "createResponse",
    "createSession",
    "forkSession",
    "getSession",
    "listResponseItems",
    "listResponses",
    "listSessions",
    "rewindSession",
    "search",
    "searchSessions",
]

# Hand-written, because OpenAPI stops at the upgrade. Audited all the same.
SOCKETS = ["watchSession"]

GENERATOR_OPTIONS = {
    "serializationLibrary": "kotlinx_serialization",
    "dateLibrary": "string",
    "enumUnknownDefaultCase": "true",
    "nonPublicApi": "true",
    "modelMutable": "false",
    "sourceFolder": "src/main/kotlin",
    "packageName": PACKAGE,
    "modelPackage": PACKAGE,
}


def operations(spec: dict) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for methods in spec["paths"].values():
        for operation in methods.values():
            if isinstance(operation, dict) and "operationId" in operation:
                found[operation["operationId"]] = operation
    return found


def audit(spec: dict) -> list[str]:
    """Report the named operations the spec does not open to a client, or does not have."""
    found = operations(spec)
    complaints = []
    for name in OPERATIONS + SOCKETS:
        operation = found.get(name)
        if operation is None:
            complaints.append(f"{name} is not in the spec")
        elif not operation.get("x-client-accessible"):
            complaints.append(f"{name} is server-side only and cannot be in a client SDK")
    return complaints


def refs(node: object) -> set[str]:
    if isinstance(node, dict):
        found = {node["$ref"]} if isinstance(node.get("$ref"), str) else set()
        for value in node.values():
            found |= refs(value)
        return found
    if isinstance(node, list):
        found = set()
        for value in node:
            found |= refs(value)
        return found
    return set()


def undefault(node: object) -> object:
    """The schema with no property defaults.

    A default is what the router does when a field is left out, which is the router's to
    apply. Generated into the model it would be the value a field starts as, and a caller's
    ``False`` equal to it would be dropped from the request while the config it was meant to
    override kept its ``True``.
    """
    if isinstance(node, dict):
        return {key: undefault(value) for key, value in node.items() if key != "default"}
    if isinstance(node, list):
        return [undefault(value) for value in node]
    return node


def prune(spec: dict) -> dict:
    """The spec with only the allowed operations and what they reach."""
    paths: dict[str, dict] = {}
    for path, methods in spec["paths"].items():
        kept = {
            method: operation
            for method, operation in methods.items()
            if isinstance(operation, dict) and operation.get("operationId") in OPERATIONS
        }
        if kept:
            paths[path] = kept

    components = spec["components"]
    keep: dict[str, set[str]] = {}
    pending = refs(paths)
    while pending:
        ref = pending.pop()
        _, _, section, name = ref.split("/")
        if name in keep.setdefault(section, set()):
            continue
        keep[section].add(name)
        pending |= refs(components[section][name])

    return {
        "openapi": spec["openapi"],
        "info": spec["info"],
        "paths": paths,
        "components": {
            section: {name: undefault(components[section][name]) for name in sorted(names)}
            for section, names in keep.items()
        },
    }


def main(argv: list[str]) -> int:
    spec = yaml.safe_load(SPEC.read_text())
    complaints = audit(spec)
    if complaints:
        for complaint in complaints:
            print(complaint, file=sys.stderr)
        return 1

    if "--check" in argv:
        print(f"{len(OPERATIONS)} client operations and a socket, none server-side only")
        return 0

    work = KOTLIN / ".codegen"
    shutil.rmtree(work, ignore_errors=True)
    (work / "out").mkdir(parents=True)
    (work / "spec.yaml").write_text(yaml.safe_dump(prune(spec), sort_keys=False))

    options = ",".join(f"{key}={value}" for key, value in GENERATOR_OPTIONS.items())
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{work}:/work",
            IMAGE, "generate",
            "-i", "/work/spec.yaml",
            "-g", "kotlin",
            "-o", "/work/out",
            "--global-property", "models,modelDocs=false,modelTests=false",
            "--additional-properties", options,
            # An open-ended object stays JSON rather than becoming Map<String, Any>, which
            # kotlinx.serialization cannot read.
            "--type-mappings", "AnyType=JsonElement",
            "--import-mappings", "JsonElement=kotlinx.serialization.json.JsonElement",
        ]
    )
    if result.returncode != 0:
        shutil.rmtree(work, ignore_errors=True)
        return result.returncode

    produced = work / "out" / "src" / "main" / "kotlin" / Path(*PACKAGE.split("."))
    shutil.rmtree(GENERATED, ignore_errors=True)
    GENERATED.mkdir(parents=True)
    for generated in sorted(produced.glob("*.kt")):
        shutil.move(str(generated), str(GENERATED / generated.name))
    shutil.rmtree(work, ignore_errors=True)

    print(f"regenerated {GENERATED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
