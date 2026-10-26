import pytest
from vision_agents.core.llm import (
    CallNotHeld,
    FunctionRegistry,
    ToolCall,
    ToolPolicy,
    calling,
    current_call,
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


class TestCurrentCall:
    async def test_outside_a_call_there_is_an_empty_one_that_does_nothing(self):
        tool_call_context = current_call()

        await tool_call_context.update("still going")
        await tool_call_context.say("one moment")
        await tool_call_context.disallow_interruptions()

        assert tool_call_context == ToolCall()

    async def test_a_call_set_for_a_function_is_the_one_it_reads(self):
        line = Line()

        async def search() -> str:
            await current_call().update("checking")
            await current_call().say("One moment.")
            return current_call().id

        with calling(ToolCall(id="call-1", line=line)):
            assert await search() == "call-1"

        assert line.sent == [
            ("call-1", "checking", False),
            ("call-1", "One moment.", True),
        ]
        assert current_call() == ToolCall()

    async def test_a_call_stopped_before_it_was_held_says_so(self):
        with calling(ToolCall(id="call-1", line=Line(holds=False))):
            with pytest.raises(CallNotHeld):
                await current_call().disallow_interruptions()
