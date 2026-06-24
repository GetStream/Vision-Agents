"""Shared utilities for OpenAI tool/function calling."""

import copy
import json
from typing import Any, Dict, List

from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema


def convert_tools_to_openai_format(
    tools: List[ToolSchema], for_realtime: bool = False
) -> List[Dict[str, Any]]:
    """Convert ToolSchema to OpenAI format.

    Args:
        tools: List of ToolSchema objects from the function registry
        for_realtime: If True, format for Realtime API (no strict field).
                      If False, format for Responses API. Strict mode is enabled
                      only for schemas where every object property is required.

    Returns:
        List of tools in OpenAI format
    """
    out = []
    for t in tools or []:
        params = copy.deepcopy(t.get("parameters_schema") or t.get("parameters") or {})
        if not isinstance(params, dict):
            params = {}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        _normalize_openai_object_schema(params)

        tool_def: Dict[str, Any] = {
            "type": "function",
            "name": t.get("name", "unnamed_tool"),
            "description": t.get("description", "") or "",
            "parameters": params,
        }

        # Realtime API does not support strict. Responses strict mode rejects
        # object schemas unless every property is listed in required, so keep
        # optional/defaulted parameters non-strict to preserve Python defaults.
        if not for_realtime:
            tool_def["strict"] = _is_strict_compatible_schema(params)

        out.append(tool_def)
    return out


def _normalize_openai_object_schema(schema: Any) -> None:
    """Normalize object schemas in-place on the OpenAI-specific schema copy."""
    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")
    is_object = schema_type == "object" or "properties" in schema
    if is_object:
        schema.setdefault("properties", {})
        schema.setdefault("additionalProperties", False)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for child in properties.values():
            _normalize_openai_object_schema(child)

    items = schema.get("items")
    if isinstance(items, dict):
        _normalize_openai_object_schema(items)

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                _normalize_openai_object_schema(variant)


def _is_strict_compatible_schema(schema: Any) -> bool:
    """Return True if every object property is listed as required."""
    if not isinstance(schema, dict):
        return True

    schema_type = schema.get("type")
    is_object = schema_type == "object" or "properties" in schema
    if is_object:
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return False

        required = schema.get("required")
        property_names = set(properties)
        if property_names:
            if not isinstance(required, list) or set(required) != property_names:
                return False
        elif "required" not in schema:
            schema["required"] = []

        if schema.get("additionalProperties") is not False:
            return False

        for child in properties.values():
            if not _is_strict_compatible_schema(child):
                return False

    items = schema.get("items")
    if isinstance(items, dict) and not _is_strict_compatible_schema(items):
        return False

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if not _is_strict_compatible_schema(variant):
                    return False

    return True


def tool_call_dedup_key(tc: NormalizedToolCallItem) -> tuple[str, str]:
    """Generate a deduplication key for a tool call.

    Uses only name and arguments, not id, so that logically identical
    tool calls are deduplicated even if they have different IDs.

    Args:
        tc: Normalized tool call item

    Returns:
        Tuple of (name, serialized_arguments) for deduplication
    """
    return (
        tc["name"],
        json.dumps(tc.get("arguments_json", {}), sort_keys=True),
    )


def parse_tool_arguments(args: str | dict) -> dict:
    """Parse tool arguments from string or dict.

    Args:
        args: Arguments as JSON string or dict

    Returns:
        Parsed arguments dict
    """
    if isinstance(args, dict):
        return args
    if not args:
        return {}
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {}
