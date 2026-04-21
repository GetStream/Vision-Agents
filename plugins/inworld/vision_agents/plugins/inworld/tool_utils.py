"""Shared utilities for Inworld tool/function calling.

Inworld's Realtime API is OpenAI-protocol-compatible, so the function/tool
schema and argument format are identical to OpenAI's Realtime API.
"""

import json

from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema


def convert_tools_to_openai_format(
    tools: list[ToolSchema], for_realtime: bool = False
) -> list[dict[str, object]]:
    """Convert ToolSchema to OpenAI format (used by Inworld Realtime).

    Args:
        tools: List of ToolSchema objects from the function registry.
        for_realtime: If True, format for Realtime API (no strict field).

    Returns:
        List of tools in OpenAI format.
    """
    out: list[dict[str, object]] = []
    for t in tools or []:
        raw_params = t.get("parameters_schema") or t.get("parameters") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
        params = {**raw_params}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        params.setdefault("additionalProperties", False)

        tool_def: dict[str, object] = {
            "type": "function",
            "name": t.get("name", "unnamed_tool"),
            "description": t.get("description", "") or "",
            "parameters": params,
        }

        if not for_realtime:
            tool_def["strict"] = True

        out.append(tool_def)
    return out


def tool_call_dedup_key(tc: NormalizedToolCallItem) -> tuple[str, str]:
    """Generate a deduplication key for a tool call."""
    return (
        tc["name"],
        json.dumps(tc.get("arguments_json", {}), sort_keys=True),
    )


def parse_tool_arguments(args: str | dict) -> dict:
    """Parse tool arguments from string or dict."""
    if isinstance(args, dict):
        return args
    if not args:
        return {}
    try:
        return json.loads(args)
    except json.JSONDecodeError:
        return {}
