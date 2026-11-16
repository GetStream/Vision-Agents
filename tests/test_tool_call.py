import pytest
from vision_agents.core.llm import (
    CallNotHeld,
    FunctionRegistry,
    ToolCallContext,
    ToolPolicy,
)


class Line:
    """A line that records what a call sent and answers holds as it is told."""

    def __init__(self, holds: bool = True):
        self.sent: list[tuple[str, str, bool]] = []
        self.holds = holds

    async def update(self, call_id: str, text: str, verbatim: bool) -> None:
        self.sent.append((call_id, text, verbatim))

    async def hold(self, call_id: str) -> bool:
        return self.holds


class TestPolicies:
    def test_a_function_registered_plainly_takes_the_defaults(self):
        registry = FunctionRegistry()

        @registry.register()
        async def weather(city: str) -> str:
            return city

        assert registry.policy("weather") == ToolPolicy()
        assert registry.policy("nothing") == ToolPolicy()

    def test_a_functions_policy_is_kept_with_it(self):
        registry = FunctionRegistry()

        @registry.register(
            interruptible=True, timeout=5, needs_approval="  Book {flight_id}?  "
        )
        async def book(flight_id: str) -> str:
            return flight_id

        assert registry.policy("book") == ToolPolicy(
            interruptible=True, timeout=5, approval="Book {flight_id}?"
        )

    def test_an_approval_naming_an_argument_the_function_lacks_is_refused(self):
        registry = FunctionRegistry()

        with pytest.raises(ValueError, match=r"\{flight\}"):

            @registry.register(needs_approval="Book {flight}?")
            async def book(flight_id: str) -> str:
                return flight_id

        assert "book" not in registry.functions

    def test_an_approval_is_checked_against_a_schema_given_as_written(self):
        registry = FunctionRegistry()
        schema = {"type": "object", "properties": {"flight_id": {"type": "string"}}}

        @registry.register(parameters_schema=schema, needs_approval="Book {flight_id}?")
        async def book(**arguments) -> str:
            return "booked"

        with pytest.raises(ValueError, match=r"\{seat\}"):

            @registry.register(
                name="seat", parameters_schema=schema, needs_approval="Seat {seat}?"
            )
            async def seat(**arguments) -> str:
                return "seated"

    @pytest.mark.parametrize(
        "options",
        [{"needs_approval": "   "}, {"timeout": -1}],
        ids=["blank", "negative"],
    )
    def test_a_policy_that_means_nothing_is_refused(self, options):
        with pytest.raises(ValueError):
            FunctionRegistry().register(**options)


class TestToolCallContext:
    async def test_a_function_whose_first_parameter_is_a_tool_call_is_handed_it(self):
        registry = FunctionRegistry()
        line = Line()

        @registry.register()
        async def search(tool_call_context: ToolCallContext, city: str) -> str:
            await tool_call_context.update("checking")
            await tool_call_context.say("One moment.")
            return f"{tool_call_context.id} {city}"

        answer = await registry.call_function(
            "search", {"city": "Rome"}, ToolCallContext(id="call-1", line=line)
        )

        assert answer == "call-1 Rome"
        assert line.sent == [
            ("call-1", "checking", False),
            ("call-1", "One moment.", True),
        ]

    def test_the_model_is_not_told_about_the_tool_call(self):
        registry = FunctionRegistry()

        @registry.register()
        async def search(tool_call_context: ToolCallContext, city: str) -> str:
            return city

        (schema,) = registry.get_tool_schemas()
        assert set(schema["parameters_schema"]["properties"]) == {"city"}
        assert schema["parameters_schema"]["required"] == ["city"]

    async def test_a_function_that_takes_only_its_arguments_is_handed_only_them(self):
        registry = FunctionRegistry()

        @registry.register()
        async def search(city: str) -> str:
            return city

        assert (
            await registry.call_function("search", {"city": "Rome"}, ToolCallContext())
            == "Rome"
        )

    async def test_without_a_call_there_is_an_empty_one_that_does_nothing(self):
        registry = FunctionRegistry()
        seen: list[ToolCallContext] = []

        @registry.register()
        async def search(tool_call_context: ToolCallContext) -> str:
            await tool_call_context.update("still going")
            await tool_call_context.disallow_interruptions()
            seen.append(tool_call_context)
            return "ok"

        assert await registry.call_function("search", {}) == "ok"
        assert seen == [ToolCallContext()]

    async def test_a_schema_given_as_written_still_hands_the_call(self):
        registry = FunctionRegistry()
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}

        @registry.register(parameters_schema=schema)
        async def search(tool_call_context: ToolCallContext, **arguments) -> str:
            return f"{tool_call_context.id} {arguments['city']}"

        assert (
            await registry.call_function(
                "search", {"city": "Rome"}, ToolCallContext(id="c")
            )
            == "c Rome"
        )

    async def test_a_call_stopped_before_it_was_held_says_so(self):
        with pytest.raises(CallNotHeld):
            await ToolCallContext(
                id="call-1", line=Line(holds=False)
            ).disallow_interruptions()
