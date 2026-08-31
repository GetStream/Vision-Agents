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

## An agent as a directory

`sync_agent("customer_support")` reads `examples/agents/customer_support/` —
`instructions.md`, `skills/*.md` and `knowledge/` — and stores them on the
backend. A hash of the directory goes with it, so a second call with the same
files does nothing.

```python
from vision_agents.plugins import stream as acceleration

await acceleration.sync_agent("customer_support")

agent = Agent(config="customer_support")
```

`Agent(config=)` fills in the edge, the remote pipeline and a phone, so the Go
backend handles routing. See `examples/agents/` for customer support, an outbound
recruiter and an inbound restaurant.

## A conversation held in writing

The same agent, with the voice left off. No call is joined, nothing is transcribed and
nothing is spoken, but everything between hearing a question and answering it is unchanged:
the same skills handed to the same slower model, and the same knowledge base looked up
mid-answer.

```python
from vision_agents.core.harness import Skill
from vision_agents.plugins import stream

config = await stream.define_agent(
    name="docs-agent",
    instructions="Answer questions about the documentation.",
    llm="llm-fast",
    subagent="llm-smart",
    skills=[Skill(name="explain", description="...", instructions="...")],
    knowledge="docs",
)

async with stream.TextSession(config_id=config.id) as session:
    async for event in session.ask("how does failover work?"):
        if event.type == "delta":
            print(event.text, end="", flush=True)
```

`define_agent` stores a named configuration in the backend's Postgres, along with the skills
it names. Both are found by name before writing, so running it again edits what is stored
rather than storing another copy. A session then names the config by id, which is how the
same agent is reached from a script, from a phone call and from anywhere else without any of
them repeating the configuration.

`TextSession.ask` streams back what the backend did on its way to an answer: `delta` as it
is written, `looked_up` when the knowledge base was searched, `delegated` and `settled`
around work handed to a skill, and `answer` when the turn is finished. Delegated work
outlives the turn that asked for it, so the model says something while it runs and the answer
arrives when it comes back.

See [example 12](../../examples/old/12_docs_agent_example) for the whole thing, including reading
this repo's markdown into a knowledge base.

A config can also be named rather than looked up by id, which is what an agent usually
wants:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=agent_user,
    llm=stream.Accelerated(config="john"),
)
```

The name is resolved when the agent joins, so the config can be defined somewhere else and
need not exist yet when the agent is built. Anything else passed to `Accelerated` overrides
what the config says.

## Calling somebody

`stream.Phone` is the telephony half of the backend, and `Agent(phone=...)` is where a call
is placed from. Stream's SIP is inbound only, so this is a vendor ringing the person and
bridging the answered leg into a Stream call; the agent is in that call before the phone
rings, so nobody answers to silence.

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=agent_user,
    llm=stream.Accelerated(config="john"),
    phone=stream.Phone(),
)

async with agent.outbound_call(from_=held, to=person, call_type="default", call_id="hello"):
    await agent.simple_response("greet the user and let them know you're a friendly AI agent")
    await agent.finish()
```

`outbound_call` also takes `ring_timeout`, `initial_digits` for reaching an extension behind
a menu, `headers` for custom SIP headers and `custom` for fields the agent can read off the
call. Vendors do not all support all of them, and one that cannot express a term refuses the
call rather than placing it without: a ring timeout that was dropped is a call sitting in
somebody's voicemail. Seven of the backend's eight implemented vendors can place a call at
all; DIDWW cannot, because it has no call control API.

See [example 13](../../examples/old/13_outbound_call_example) for the whole thing.

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
