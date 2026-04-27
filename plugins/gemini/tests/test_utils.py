from vision_agents.plugins.gemini.utils import convert_tools_to_provider_format


class TestConvertToolsToProviderFormat:
    def test_routes_mcp_schema_to_parameters_json_schema(self):
        tools = [
            {
                "name": "search_docs",
                "description": "Search knowledge base",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                    "$schema": "http://json-schema.org/draft-07/schema#",
                },
            }
        ]

        result = convert_tools_to_provider_format(tools)

        decl = result[0]["function_declarations"][0]
        assert "parameters" not in decl
        schema = decl["parameters_json_schema"]
        assert "$schema" not in schema
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["type"] == "string"

    def test_strips_nested_schema_meta(self):
        tools = [
            {
                "name": "nested",
                "description": "",
                "parameters_schema": {
                    "type": "object",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "inner": {
                            "type": "object",
                            "$schema": "http://json-schema.org/draft-07/schema#",
                        }
                    },
                },
            }
        ]

        schema = convert_tools_to_provider_format(tools)[0]["function_declarations"][0][
            "parameters_json_schema"
        ]
        assert "$schema" not in schema
        assert "$schema" not in schema["properties"]["inner"]
