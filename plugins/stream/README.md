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

`agents/customer_support/agent.yaml` is what makes a directory an agent, and
`instructions.md`, `skills/*.md` and `knowledge/` are what it holds. Joining a call
stores them on the backend, so naming the config is the whole of it:

```python
from vision_agents.core import Agent

agent = Agent(config="customer_support")
```

`agent.yaml` also says what the agent runs on, so the models are decided on disk with
everything else:

```yaml
name: customer_support
description: Support for a subscription business.
mode: voice
llm: llm-fast
subagent: llm-thinking
stt: stt-fast
tts: tts-fast
voice: aurora
search: search-fast
greeting: Thanks for calling, how can I help?
sandbox: daytona
plugins:
  - gmail
keyterms:
  - Vision Agents
tags:
  project: support
```

Everything but the name is optional, and a setting the file leaves out leaves whatever is
stored, so a model chosen in the dashboard survives a sync that says nothing about it. A key
nobody knows is refused rather than dropped: a misspelled `llm` that went quietly would
leave the agent running on a model the file does not name.

`.agent_sync` next to `agent.yaml` records the md5 of what was last stored, so a run that
changed nothing costs a file read rather than a request. `sync_agent("customer_support")`
is still there for storing a directory ahead of time, without joining anything.

`Agent(config=)` fills in the edge, the remote pipeline and a phone, so the Go
backend handles routing. See `examples/voice_agents/` for customer support, an outbound
recruiter and an inbound restaurant.

## A conversation held in writing

The same agent, with the voice left off. No call is joined, nothing is transcribed and
nothing is spoken, but everything between hearing a question and answering it is unchanged:
the same skills handed to the same slower model, and the same knowledge base looked up
mid-answer.

```python
from vision_agents.core import Agent
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

agent = Agent(config=config.name)
async with agent.chat():
    async for event in agent.ask("how does failover work?"):
        if event.type == "agent_speech_delta":
            print(event.text, end="", flush=True)
```

`define_agent` stores a named configuration in the backend's Postgres, along with the skills
it names. Both are found by name before writing, so running it again edits what is stored
rather than storing another copy. An agent then names the config, which is how the same
agent is reached from a script, from a phone call and from anywhere else without any of them
repeating the configuration.

`agent.ask` streams back what the backend did on its way to an answer: `agent_speech_delta`
as it is written, `looked_up` when the knowledge base was searched, `delegated` and
`task_settled` around work handed to a skill, and `agent_speech` when the turn is finished.
Delegated work outlives the turn that asked for it, so the model says something while it runs
and the answer arrives when it comes back.

`agent.chat(agent_id)` answers in a conversation that already exists rather than in one of
the agent's own, which is how a message written to a channel is answered where it was asked.

See [the docs agent](../../examples/text_agents/docs_agent) for the whole thing, including
reading this repo's markdown into a knowledge base.

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
    await agent.responses.create("greet the user and let them know you're a friendly AI agent")
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

`Router().resolve("sonic_36")` asks the backend which kind of model a name is and returns the
plugin for it. That costs a request at startup, so naming the modality is better when you
know it.

## A router config, four modalities

`stream.Router` is the same routing with the options said once. It names a stored router
config, and every keyword on a call overrides one field of it:

```python
from vision_agents.plugins.stream import Router, define_router

await define_router(
    "healthcare",
    stt={"diarize": True, "keyterms": ["metformin", "sertraline"]},
    tts={"voice": "dc4e4a1f", "speed": 1.1},
    search={"include_domains": ["pubmed.ncbi.nlm.nih.gov"]},
)

router = Router("healthcare")

async with router.stt.realtime() as stt:
    ...

agent = Agent(edge=getstream.Edge(), agent_user=agent_user, stt=router.stt.realtime())
```

`realtime()` returns a configured, not-yet-started session: `async with` starts and closes
it, and handing the same object to an `Agent` lets the agent own its lifecycle instead.

`recording()` is the non-realtime form — a whole source in, a whole result out, served by the
batch half of a vendor rather than the streaming one, which is both cheaper and more accurate:

```python
transcript = await router.stt.recording("interview.mp4", diarize=True, words=True)
print(transcript.text, transcript.speakers)

audiobook = await router.tts.recording(chapter, format="mp3_44100_128")
hits = await router.search("perioperative antibiotic guidance", results=5)
```

A recording takes a URL, a path or the bytes, and waits for the job unless you pass a
`callback`, in which case it returns as soon as the job is accepted.

An option a modality does not have is refused rather than sent and ignored, and so is one no
provider behind that target can express: a transcript that was quietly not diarized is worse
than being told. What each provider can express is in the `router-stt`, `router-tts`,
`router-llm` and `router-search` skills.

## Images and delegated vision

Keep the conversation on `llm-fast` and bind visual analysis to a separate worker in
`agent.yaml`:

```yaml
llm: llm-fast
subagents:
  default: llm-thinking
  vision: vlm
video:
  max_frames: 1
```

The built-in `vision` skill captures evidence when asked and runs on the `vision` worker.
Worker preparation, frame encoding and inference run asynchronously, so the conversation
can continue. Custom skills opt in with `subagent: vision` and `capture_video: true`.
`subagent: llm-thinking` remains shorthand for `subagents.default`; declare only one form
of the default in a configuration layer. Named overrides merge by key; an empty target
removes that worker. `stream.define_agent` also accepts `subagents`, `video_source` and
`video_max_frames`.

Explicit attachments use the same worker in an accelerated agent:

```python
from vision_agents.core.llm import ImageContent

await agent.responses.create(
    "Compare these pictures.",
    images=[ImageContent(data=first_jpeg), ImageContent(url=second_url)],
)
```

The main conversation receives the question, pending task and eventual findings. The
vision worker receives the images. For standalone inference, use
`stream.LLM(target="vlm").responses.create(...)` and iterate the response stream. The
standalone API also accepts an ordered list of strings and `ImageContent` as its first
argument to preserve text/image interleaving. With an attached conversation, images remain
on their original turns for follow-up questions and are released when those turns are
removed from history. Image URLs are HTTP(S) URLs or bytes encoded
by the SDK; the wire format is `{"type":"image_url","image_url":{"url":"…","detail":"auto"}}`.
Image detail accepts `auto`, `low` or `high`. `vlm` requires image capability before model
selection; `llm-fast` retains its existing routing policy.

Camera evidence uses local receive timestamps in Unix milliseconds; backend and video
worker system clocks must be synchronized. Each observation
buffer retains at most 64 frames and 32 MiB. Capture selects 1–8 frames at or before task
acceptance, pins them for that task, and rejects missing or stale evidence. The newest
selected frame must be no more than five seconds old. Multiple available cameras require
an explicit source; missing or ambiguous evidence produces a clarification question.
`video.source` can name a raw `participant/track` or a processor source such as
`roboflow_streaming`. A vision directive can override selection for one question, for
example `<ask skill="vision" frames="2">What changed?</ask>`.

Roboflow continues processing independently. Its observations retain raw frames and
separately timestamped predictions; unavailable prediction/frame alignment is stated
explicitly. See the [flower spotter](../../examples/video_agents/flower_spotter) and
[standalone image example](../../examples/video_agents/describe_image).

On the session socket, `get_video_frames` is a reserved capture request carrying a task
ID, source, acceptance timestamp and limit. The matching `tool_result` carries ordered
metadata/image parts. A `tool_cancel` event with that request ID cancels capture or a
running local tool; late results are discarded. Existing native speech-to-speech video
forwarding is unchanged. Integrating those engines with the shared task/result lifecycle
is a separate milestone.

## Regenerating the client

`_generated/` comes from `acceleration/api/openapi.yaml` and is committed. After changing
the spec:

```bash
uv run plugins/stream/generate.py
```

The WebSockets are hand-written in `_socket.py`, since OpenAPI stops at the upgrade.
