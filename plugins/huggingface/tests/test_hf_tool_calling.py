"""Tests for the shared _hf_tool_calling helpers and tool call loop."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from vision_agents.plugins.huggingface._hf_tool_calling import (
    accumulate_tool_call_chunk,
    convert_tools_to_hf_format,
    extract_tool_calls_from_hf_response,
    finalize_pending_tool_calls,
)
from vision_agents.plugins.huggingface._tool_call_loop import (
    run_tool_call_loop,
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


class TestRunToolCallLoop:
    async def test_multi_round_tool_calls(self):
        """Follow-up that returns more tool calls triggers another round."""
        executed = []

        async def fake_call_function(name: str, arguments: dict) -> str:
            executed.append(name)
            return f"{name}_result"

        llm = MagicMock()
        llm._dedup_and_execute = AsyncMock(
            side_effect=[
                (
                    [
                        (
                            {"id": "c1", "name": "tool_a", "arguments_json": {}},
                            "tool_a_result",
                            None,
                        )
                    ],
                    {("c1", "tool_a", "{}")},
                ),
                (
                    [
                        (
                            {"id": "c2", "name": "tool_b", "arguments_json": {}},
                            "tool_b_result",
                            None,
                        )
                    ],
                    {("c1", "tool_a", "{}"), ("c2", "tool_b", "{}")},
                ),
            ]
        )
        llm._sanitize_tool_output = MagicMock(side_effect=lambda v: str(v))

        from vision_agents.core.llm.llm import LLMResponseEvent

        round1_response = LLMResponseEvent(original=None, text="round1")
        round2_response = LLMResponseEvent(original=None, text="final answer")

        call_count = 0

        async def generate_followup(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                next_calls = [
                    {
                        "type": "tool_call",
                        "id": "c2",
                        "name": "tool_b",
                        "arguments_json": {},
                    }
                ]
                return round1_response, next_calls
            return round2_response, []

        initial_tool_calls = [
            {
                "type": "tool_call",
                "id": "c1",
                "name": "tool_a",
                "arguments_json": {},
            }
        ]

        result = await run_tool_call_loop(
            llm,
            initial_tool_calls,
            [{"role": "user", "content": "test"}],
            generate_followup,
        )

        assert result.text == "final answer"
        assert call_count == 2
        assert llm._dedup_and_execute.call_count == 2

    async def test_stops_when_no_tool_calls_in_followup(self):
        """Loop stops after first round when follow-up has no tool calls."""
        llm = MagicMock()
        llm._dedup_and_execute = AsyncMock(
            return_value=(
                [
                    (
                        {"id": "c1", "name": "fn", "arguments_json": {}},
                        "ok",
                        None,
                    )
                ],
                {("c1", "fn", "{}")},
            )
        )
        llm._sanitize_tool_output = MagicMock(side_effect=lambda v: str(v))

        from vision_agents.core.llm.llm import LLMResponseEvent

        final = LLMResponseEvent(original=None, text="done")

        async def generate_followup(messages):
            return final, []

        initial = [
            {"type": "tool_call", "id": "c1", "name": "fn", "arguments_json": {}}
        ]

        result = await run_tool_call_loop(
            llm, initial, [{"role": "user", "content": "q"}], generate_followup
        )

        assert result.text == "done"
        assert llm._dedup_and_execute.call_count == 1
