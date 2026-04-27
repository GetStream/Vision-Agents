from typing import Any, Dict, List

from vision_agents.core.llm.llm_types import ToolSchema


_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset({"$schema"})


def _strip_unsupported_schema_keys(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_unsupported_schema_keys(v)
            for k, v in node.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_strip_unsupported_schema_keys(item) for item in node]
    return node


def convert_tools_to_provider_format(tools: List[ToolSchema]) -> List[Dict[str, Any]]:
    """
    Convert ToolSchema objects to Gemini format.
    Args:
        tools: List of ToolSchema objects
    Returns:
        List of tools in Gemini format
    """
    function_declarations = []
    for tool in tools:
        function_declarations.append(
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters_json_schema": _strip_unsupported_schema_keys(
                    tool["parameters_schema"]
                ),
            }
        )

    # Return as dict with function_declarations (SDK accepts dicts)
    return [{"function_declarations": function_declarations}]
