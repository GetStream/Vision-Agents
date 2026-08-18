# Stream acceleration plugin

Runs the voice pipeline in the Go acceleration backend instead of in Python. The backend
joins the call, hears the caller, answers and speaks; Python configures it and runs the
functions the model asks for.

```bash
uv add vision-agents-plugins-stream
```

Two environment variables point at the backend:

```bash
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=acme
```

## The whole pipeline, remotely

```python
from vision_agents.core import Agent
from vision_agents.core.harness import Daytona, DefaultHarness
from vision_agents.plugins import getstream, stream

agent = Agent(
    edge=getstream.Edge(),
    agent_user=agent_user,
    instructions="Keep your replies short.",
    llm=stream.Accelerated(model="gemma4", stt="realtime-best", tts="sonic_36"),
    harness=DefaultHarness(use_skills=True, subagents={"default": "llm-smart"}, vm=Daytona),
    cost_tracking={"project": "moderation", "environment": "dev"},
    memory_filter={"user_id": "222", "company_id": "12312"},
)


@agent.llm.register_function(description="The weather where the caller is")
async def weather(city: str) -> str:
    return f"It is raining in {city}."
```

`register_function` works as it does with any other LLM. The model asks for the function
over the session's socket, this plugin runs it here and sends back what it returned.

`harness`, `cost_tracking` and `memory_filter` are configuration rather than behaviour: they
are serialized into the session and acted on by the backend. `memory_filter["user_id"]` is
who the memories are about, and everything else in it narrows recall further.

The agent's transcripts, conversation and events all work as they do locally, because the
events the backend sends back are recorded into the same places.

`vm=Daytona` gives the subagent somewhere to run code it writes, and needs `DAYTONA_API_KEY`
on the backend. Only the subagent is offered it: running code takes seconds, and the model
holding the conversation has none to spare.

## One modality at a time

For a pipeline that stays in Python, each modality can be routed on its own. Failover and
cost tracking work the same way, because it is the same router:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=agent_user,
    stt=stream.STT("en-low-latency"),
    llm=stream.LLM("llm-fast"),
    tts=stream.TTS("sonic_36", voice="dc4e4a1f"),
)
```

`stream.Router("sonic_36")` asks the backend which kind of model a name is and returns the
plugin for it. That costs a request at startup, so naming the modality is better when you
know it.

## Regenerating the client

`_generated/` comes from `acceleration/api/openapi.yaml` and is committed. After changing
the spec:

```bash
uv run plugins/stream/generate.py
```

The WebSockets are hand-written in `_socket.py`, since OpenAPI stops at the upgrade.
