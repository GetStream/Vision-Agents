"""Tests for the shared _hf_tool_calling helpers."""

from types import SimpleNamespace

from vision_agents.plugins.huggingface._hf_tool_calling import (
    accumulate_tool_call_chunk,
    convert_tools_to_hf_format,
    extract_tool_calls_from_hf_response,
    finalize_pending_tool_calls,
)


def _tc_chunk(
    index: int, tc_id: str | None, name: str | None, arguments: str | None
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestConvertToolsToHFFormat:
    async def test_basic_conversion(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        result = convert_tools_to_hf_format(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather"
        assert (
            result[0]["function"]["parameters"]["properties"]["city"]["type"]
            == "string"
        )

    async def test_empty_tools(self):
        assert convert_tools_to_hf_format([]) == []
        assert convert_tools_to_hf_format(None) == []

    async def test_missing_fields_use_defaults(self):
        tools = [{"name": "f"}]
        result = convert_tools_to_hf_format(tools)
        assert result[0]["function"]["description"] == ""
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert result[0]["function"]["parameters"]["properties"] == {}


class TestStreamingToolCallAccumulation:
    async def test_single_chunk_tool_call(self):
        pending: dict = {}
        accumulate_tool_call_chunk(
            pending, _tc_chunk(0, "call-1", "get_weather", '{"city": "SF"}')
        )
        result = finalize_pending_tool_calls(pending)

        assert len(result) == 1
        assert result[0]["id"] == "call-1"
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments_json"] == {"city": "SF"}

    async def test_multi_chunk_arguments(self):
        pending: dict = {}
        accumulate_tool_call_chunk(
            pending, _tc_chunk(0, "call-1", "search", '{"query":')
        )
        accumulate_tool_call_chunk(pending, _tc_chunk(0, None, None, ' "hello"}'))
        result = finalize_pending_tool_calls(pending)

        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["arguments_json"] == {"query": "hello"}

    async def test_multiple_parallel_tool_calls(self):
        pending: dict = {}
        accumulate_tool_call_chunk(pending, _tc_chunk(0, "call-a", "tool_a", "{}"))
        accumulate_tool_call_chunk(pending, _tc_chunk(1, "call-b", "tool_b", "{}"))
        result = finalize_pending_tool_calls(pending)

        assert len(result) == 2
        names = {tc["name"] for tc in result}
        assert names == {"tool_a", "tool_b"}

    async def test_malformed_json_arguments(self):
        pending: dict = {}
        accumulate_tool_call_chunk(pending, _tc_chunk(0, "call-1", "f", "not json"))
        result = finalize_pending_tool_calls(pending)

        assert result[0]["arguments_json"] == {}


class TestExtractToolCallsFromHFResponse:
    async def test_response_with_tool_calls(self):
        tc = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="get_weather", arguments='{"city": "SF"}'),
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tc]))]
        )

        result = extract_tool_calls_from_hf_response(response)
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments_json"] == {"city": "SF"}

    async def test_response_without_tool_calls(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))]
        )
        assert extract_tool_calls_from_hf_response(response) == []

    async def test_empty_choices(self):
        response = SimpleNamespace(choices=[])
        assert extract_tool_calls_from_hf_response(response) == []
