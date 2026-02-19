# Testing Vision-Agents

Test your agent's behavior in pure text mode — no audio, video, or edge connection needed.

You write a conversation scenario, the framework runs it against a real LLM, and you verify the results: what the agent said, which tools it called, and whether the response makes sense.

## Setup

You need two LLM instances:

- **`llm`** — the LLM your agent uses (with tools registered)
- **`judge`** — a separate LLM that evaluates whether responses match your expectations

```python
from vision_agents.plugins import gemini
from vision_agents.testing import TestEval

llm = gemini.LLM("gemini-2.5-flash-lite")
judge_llm = gemini.LLM("gemini-2.5-flash-lite")
```

> Why a separate judge? The judge makes its own LLM calls to evaluate responses. Using the same instance would mix judge prompts into the agent's conversation history.

## Writing Your First Test

```python
async def test_greeting():
    llm = gemini.LLM("gemini-2.5-flash-lite")
    judge_llm = gemini.LLM("gemini-2.5-flash-lite")

    async with TestEval(llm=llm, judge=judge_llm, instructions="Be friendly") as session:
        response = await session.simple_response("Hello!")
        await response.judge(intent="Responds with a friendly greeting")
        response.no_more_events()
```

What's happening here:

1. **`TestEval(...)`** creates a test session with your LLM and instructions
2. **`simple_response("Hello!")`** sends "Hello!" to the LLM and captures everything that happens
3. **`judge(intent="...")`** checks that the agent's reply matches your intent
4. **`no_more_events()`** confirms there's nothing unexpected left (no extra tool calls, no extra messages)

## Testing Tool Calls

When your agent uses tools, you can verify it called the right function with the right arguments:

```python
async def test_weather():
    llm = gemini.LLM("gemini-2.5-flash-lite")
    judge_llm = gemini.LLM("gemini-2.5-flash-lite")

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> dict:
        return {"temperature": 22, "condition": "sunny"}

    async with TestEval(
        llm=llm,
        judge=judge_llm,
        instructions="Use the get_weather tool when asked about weather.",
    ) as session:
        response = await session.simple_response("What's the weather in Tokyo?")

        # Verify the agent called the right tool with the right arguments
        response.function_called("get_weather", arguments={"location": "Tokyo"})

        # Verify the agent gave a useful answer based on the tool result
        await response.judge(intent="Reports weather for Tokyo including temperature")

        # Confirm nothing unexpected happened
        response.no_more_events()
```

### Partial argument matching

You don't have to check every argument. Only the keys you specify are compared:

```python
# Passes even if the actual arguments also include "limit" and "offset"
response.function_called("search", arguments={"query": "hello"})
```

### Checking tool output

Normally you don't need to inspect tool outputs — `function_called()` automatically skips past them. But if you want to verify what a tool returned:

```python
response.function_output(output={"temperature": 22, "condition": "sunny"})
response.function_output(is_error=True)  # verify the tool errored
```

### Multiple tool calls

Just call `function_called()` for each one, in order:

```python
response = await session.simple_response("Compare weather in Tokyo and Berlin")
response.function_called("get_weather", arguments={"location": "Tokyo"})
response.function_called("get_weather", arguments={"location": "Berlin"})
await response.judge(intent="Compares weather in both cities")
response.no_more_events()
```

## Testing with Mock Tools

You can replace tool implementations to control what they return — useful for testing error handling or specific scenarios:

```python
from vision_agents.testing import mock_tools

async def test_tool_error():
    # ... setup llm with get_weather registered ...

    async with TestEval(llm=llm, judge=judge_llm, instructions="...") as session:
        # Make get_weather throw an error
        with mock_tools(session.llm, {
            "get_weather": lambda location: (_ for _ in ()).throw(
                RuntimeError("Service unavailable")
            ),
        }):
            response = await session.simple_response("What's the weather in Paris?")

        # Verify the agent handled the error gracefully
        await response.judge(
            intent="Informs the user that it could not get the weather"
        )
```

The tool schemas (name, description, parameters) stay visible to the LLM — only the callable is swapped. Originals are always restored when the `with` block exits.

## Multi-Turn Conversations

Conversation history is preserved between `simple_response()` calls, so you can test context retention:

```python
async def test_remembers_name():
    async with TestEval(llm=llm, judge=judge_llm, instructions="Be helpful") as session:
        response = await session.simple_response("My name is Alice")
        await response.judge(intent="Acknowledges the user's name")

        response = await session.simple_response("What's my name?")
        await response.judge(intent="Correctly recalls the name Alice")
```

## Testing Grounding (No Hallucination)

Verify the agent doesn't make things up:

```python
async def test_doesnt_hallucinate():
    async with TestEval(llm=llm, judge=judge_llm, instructions="Be helpful") as session:
        response = await session.simple_response("What city was I born in?")
        await response.judge(
            intent="Does NOT claim to know the user's birthplace. "
            "Asks for clarification or says it doesn't have that info."
        )
```

## Inspecting the Response

`simple_response()` returns a `TestResponse` with all the data you need for custom checks:

```python
response = await session.simple_response("What's the weather?")

response.input            # "What's the weather?" — what you sent
response.output           # "It's sunny, 22°C" — what the agent replied
response.function_calls   # [FunctionCallEvent(...)] — which tools were called
response.events           # full chronological event list
response.duration_ms      # how long the turn took (ms)
```

You can use these fields for any custom assertions beyond the built-in helpers:

```python
assert response.output is not None
assert len(response.function_calls) == 1
assert response.duration_ms < 5000
```

## Running Tests

```bash
# Integration tests (requires GOOGLE_API_KEY in .env)
uv run py.test path/to/test_file.py -m integration

# See every event and judge result
VISION_AGENTS_EVALS_VERBOSE=1 uv run py.test path/to/test_file.py -m integration

# With debugger
uv run py.test path/to/test_file.py -m integration -s --timeout=0 --pdb
```

## Recommended Project Structure

Separate your LLM setup from Agent construction so tests can reuse the LLM without needing Edge/STT/TTS infrastructure:

```python
# my_agent.py

INSTRUCTIONS = "You're a helpful voice assistant. Be concise."

def setup_llm(model: str = "gemini-2.5-flash-lite") -> gemini.LLM:
    llm = gemini.LLM(model)

    @llm.register_function(description="Get current weather")
    async def get_weather(location: str) -> dict:
        return await get_weather_by_location(location)

    return llm
```

```python
# test_my_agent.py

from .my_agent import INSTRUCTIONS, setup_llm

async def test_weather():
    llm = setup_llm()
    judge_llm = gemini.LLM("gemini-2.5-flash-lite")

    async with TestEval(llm=llm, judge=judge_llm, instructions=INSTRUCTIONS) as session:
        response = await session.simple_response("What's the weather in Berlin?")
        response.function_called("get_weather", arguments={"location": "Berlin"})
        await response.judge(intent="Reports weather for Berlin")
        response.no_more_events()
```

## Reference

### TestEval

| | |
|---|---|
| `TestEval(llm, instructions, judge)` | Create a test session |
| `await session.simple_response(text)` | Send text, get `TestResponse` |
| `session.llm` | Access the LLM instance |

### TestResponse — data

| Field | Type | Description |
|---|---|---|
| `input` | `str` | What the user said |
| `output` | `str \| None` | What the agent replied |
| `events` | `list[RunEvent]` | Full event list |
| `function_calls` | `list[FunctionCallEvent]` | Which tools were called |
| `duration_ms` | `float` | Turn duration (ms) |

### TestResponse — assertions

| Method | What it verifies |
|---|---|
| `function_called(name?, arguments?)` | Agent called the expected tool |
| `function_output(output?, is_error?)` | Tool returned the expected result |
| `await judge(intent?)` | Agent's reply matches the intent |
| `no_more_events()` | Nothing unexpected happened |

### mock_tools

| | |
|---|---|
| `mock_tools(llm, {"name": callable})` | Temporarily replace tool implementations |

### Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | API key for Gemini LLM |
| `VISION_AGENTS_TEST_MODEL` | Override test model (default: `gemini-2.5-flash-lite`) |
| `VISION_AGENTS_EVALS_VERBOSE` | Set to `1` for detailed output during test runs |
