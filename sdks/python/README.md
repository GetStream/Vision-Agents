# Vision Agents Python SDK

```bash
uv add vision-agents
```

The examples below are cut down to the part worth reading. Complete, runnable versions of
each live in [examples/agents](https://github.com/GetStream/Vision-Agents/tree/main/examples/agents).

## Initializing the agent

Name a stored config and the acceleration backend runs the whole pipeline: it joins the
call, transcribes, answers and speaks. `edge`, `llm`, `agent_user` and `phone` all come from
the config, so an agent is a name:

```python
agent = Agent(config="simple_voice_ai")
```

## Skills, knowledge and instructions as a directory

An agent can be written down as a directory:

```
examples/agents/customer_support/
  instructions.md        the system prompt
  skills/refund.md       frontmatter (name, description, deadline) and a body
  knowledge/policy.md    what the agent may look things up in
```

`sync_agent` pushes it to the acceleration backend and stores a config named after the
directory. It sends a hash of the files with it, so a second run over the same files does
nothing:

```python
await acceleration.sync_agent("customer_support")

agent = Agent(config="customer_support")
```

## A response in writing

No call, nothing transcribed and nothing spoken, but the same instructions, skills and
knowledge base a call would have had:

```python
async with acceleration.TextSession(config_id="customer_support") as session:
    async for event in session.ask("how do refunds work?"):
        if event.type == "delta":
            print(event.text, end="", flush=True)
```

## Voice

`join` puts the agent in a Stream call and `simple_response` asks it to say something.
Leaving the block waits for the call to end, so there is nothing to wait on by hand:

```python
agent = Agent(config="simple_voice_ai")

call = await agent.create_call("default", "my-call")
async with agent.join(call):
    await agent.simple_response("greet the user in one short sentence")
```

Pass `wait_for_end=False` to hang up as soon as the block is done instead.

## Inbound and outbound calls

Ringing somebody. The agent is in the call before the phone rings, so nobody answers to
silence:

```python
async with agent.outbound_call(from_="+15125551234", to="+15125555678"):
    await agent.simple_response("greet the user and say you're an AI agent")
```

Answering. The caller is already in the call, so `answer` attaches to theirs rather than
making one, and dispatch runs the handler once per call:

```python
dispatch = acceleration.StreamDispatch()


@dispatch.wait_for_call()
async def answer(call: InboundCall) -> None:
    agent = Agent(config="john")
    async with agent.answer(call):
        await agent.simple_response("greet the caller and ask how you can help")
```

Both need a router with a telephony vendor configured and a number bought from it, and
`outbound_call` needs an agent built with `phone=acceleration.Phone()` (a config fills that
in). See [acceleration/README.md](https://github.com/GetStream/Vision-Agents/blob/main/acceleration/README.md).

## Routing one modality at a time

A pipeline can stay in Python and route one modality at a time. The turns, the model and the
conversation are here; only the transcribing happens elsewhere, over a socket, with the same
failover and cost tracking a full session gets:

```python
agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Jean", id="agent"),
    stt=acceleration.STT("en-low-latency", language="en", tags={"project": "demo"}),
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=elevenlabs.TTS(),
)
```

`"en-low-latency"` is a capability shortcut: the router picks a provider and fails over to
another if one is down. A concrete `"deepgram/flux-general-en"` pins it to one model instead.
`acceleration.LLM` and `acceleration.TTS` route the other two modalities the same way.

`acceleration.Router` says the options once instead, as a stored config with four namespaces.
Each of the three streaming modalities has a `realtime()` session and a `recording()` job, and
search is one round trip:

```python
router = acceleration.Router("healthcare")

async with router.stt.realtime() as stt:
    ...

transcript = await router.stt.recording("interview.mp4", diarize=True)
audiobook = await router.tts.recording(chapter, format="mp3_44100_128")
hits = await router.search("perioperative antibiotic guidance", results=5)
```

A recording is served by the batch half of a vendor rather than the streaming one, which is
cheaper and more accurate. Every keyword overrides one field of the config, and an option no
provider behind the target can express is refused rather than dropped: a transcript that was
quietly not diarized is worse than being told.

# Open Vision Agents by Stream

Build Vision Agents quickly with any model or video provider.

-  **Video AI**: Built for real-time video AI. Combine Yolo, Roboflow and others with gemini/openai realtime
-  **Low Latency**: Join quickly (500ms) and low audio/video latency (30ms)
-  **Open**: Built by Stream, but use any video edge network that you like
-  **Native APIs**: Native SDK methods from OpenAI (create response), Gemini (generate) and Claude (create message). So you're never behind on the latest features
-  **SDKs**: SDKs for React, Android, iOS, Flutter, React, React Native and Unity.

Created by Stream, uses [Stream's edge network](https://getstream.io/video/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents) for ultra-low latency.

See [Github](https://github.com/GetStream/Vision-Agents).