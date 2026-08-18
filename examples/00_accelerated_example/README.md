# Accelerated Agent Example

The same voice agent as [example 01](../01_simple_agent_example), with the pipeline running
in the Go acceleration backend instead of in Python.

Python creates the call, configures the agent and runs the functions the model asks for.
The backend joins the call, transcribes what it hears, answers it and speaks back, routing
each modality across providers and failing over between them.

## Prerequisites

- Python 3.13 or higher
- A running acceleration router (see [acceleration/README.md](../../acceleration/README.md))
- API keys for:
    - [Stream](https://getstream.io/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents) (for video/audio infrastructure)
    - Whatever providers the router is configured with

## Installation

1. Go to the example's directory

    ```bash
    cd examples/00_accelerated_example
    ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

3. Create a `.env` file:

   ```
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   STREAM_ACCELERATION_URL=http://localhost:8080
   STREAM_ACCELERATION_CUSTOMER_ID=examples
   ```

## Running the Example

```bash
uv run accelerated_example.py run
```

## What is different

Only the LLM:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="My accelerated AI friend", id="agent"),
    instructions=INSTRUCTIONS,
    llm=stream.Accelerated(model="llm-fast", stt="en-low-latency", tts="en-low-latency"),
    harness=DefaultHarness(use_skills=True, subagents={"default": "llm-smart"}, vm=Daytona),
    cost_tracking={"project": "examples", "environment": "dev"},
    memory_filter={"user_id": "222", "company_id": "12312"},
)
```

There is no `stt`, `tts` or `turn_detection` on the agent, because none of that happens
here. Each target is a `provider/model` name or a capability shortcut such as `llm-fast`,
and the router picks between the providers that can serve it.

**`harness`** is what stands between what the caller said and the model that answers them.
The fast model holds the conversation; work it should not answer itself is handed to a
slower one, which can run code in a sandbox. It is configuration: the loop runs in Go.

**`cost_tracking`** labels every request the call makes, so spend can be attributed to more
than a model name.

**`memory_filter`** is who the memories are about. `user_id` is the identity recall is keyed
by, and every other key narrows it further.

### Function calling stays in Python

```python
@agent.llm.register_function(description="Get current weather for a location")
async def get_weather(location: str) -> Dict[str, Any]:
    return await get_weather_by_location(location)
```

The function is declared to the backend when the session is created. When the model reaches
for it, the call arrives over the session's socket, runs here, and the result goes back
while the model is still waiting. Nothing about writing a function changes.

### Routing one modality at a time

If you would rather keep the pipeline in Python and only route the models, each modality
has its own plugin:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=agent_user,
    stt=stream.STT("en-low-latency"),
    llm=stream.LLM("llm-fast"),
    tts=stream.TTS("sonic_36", voice="dc4e4a1f"),
)
```

## Learn More

- [The acceleration backend](../../acceleration/README.md)
- [The stream plugin](../../plugins/stream/README.md)
- [Main Vision Agents README](../../README.md)
