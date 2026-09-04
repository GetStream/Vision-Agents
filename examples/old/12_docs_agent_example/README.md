# Docs Agent Example

An agent that answers questions about the documentation, in writing. No call, no
microphone, no speaker: just text in and text out.

What it keeps is everything between hearing a question and answering it. A fast model
holds the conversation, looks the question up in a knowledge base filled from this repo's
markdown, and hands anything that deserves more than a fast model to a slower one through
a skill. That is the same agent a phone call would get, minus the voice.

## Prerequisites

- Python 3.13 or higher
- A running acceleration router (see [acceleration/README.md](../../../acceleration/README.md))
  with `TURBOPUFFER_API_KEY` set, since the agent looks things up
- Go, to fill the knowledge base once

## Installation

1. Go to the example's directory

    ```bash
    cd examples/old/12_docs_agent_example
    ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

3. Create a `.env` file:

   ```
   STREAM_ACCELERATION_URL=http://localhost:8080
   STREAM_ACCELERATION_CUSTOMER_ID=examples
   ```

## Filling the knowledge base

The agent can only answer from what it has read, so read the docs into it first:

```bash
cd ../../../acceleration
go run ./cmd/knowledge -namespace docs ../docs ../README.md ../sdks/python
```

Markdown is cut at its headings, long sections are cut again at paragraph breaks, and each
passage keeps the heading it was found under so it still makes sense on its own. Passages
are keyed by where they came from, so running it again after editing a file replaces that
file's passages rather than adding a second copy of them.

## Running the Example

```bash
uv run docs_agent.py
```

```
> what does the harness do?

  [read 4 passages about 'harness']
Let me pull that together for you.

  [handed to explain: what the harness is and how skills reach the subagent]

  [explain came back]
The harness is what stands between what a caller said and the model that answers them...
```

## How it works

### The agent is configured once, in Postgres

```python
config = await stream.define_agent(
    name="docs-agent",
    instructions=INSTRUCTIONS,
    llm="llm-fast",
    subagent="llm-smart",
    skills=SKILLS,
    knowledge="docs",
)
```

An agent config is a stored, named configuration: which models answer, what the agent is
told to be, which skills it may hand work to and which knowledge base it may read. Sessions
name it by id, so the same agent can be reached from this script, from a phone call or from
the dashboard without any of them repeating the configuration.

`define_agent` finds the config and its skills by name before writing, so running the
example twice edits what is stored rather than storing another copy of it.

Without a `subagent` the fast model answers everything itself and the skills mean nothing,
because there is nobody to hand the work to.

### Skills are prompts for a slower model

```python
Skill(
    name="explain",
    description="a part of the documentation the reader has asked to understand",
    deadline_seconds=25,
    instructions="You are the explaining half of a documentation agent...",
)
```

There is nothing behind a skill but a better model and more time. The `description` is the
one line the fast model chooses by; the `instructions` are the full prompt, which only the
subagent sees. A skill defined here replaces the built-in of the same name, which is how
`explain` becomes an explanation written to be read rather than one written to be heard on
a phone.

Delegated work outlives the turn that asked for it: the fast model says something while it
runs, and the answer arrives on the socket when it comes back.

### The conversation is held in writing

```python
async with stream.TextSession(config_id=config.id) as session:
    async for event in session.ask("how does routing fail over?"):
        if event.type == "delta":
            print(event.text, end="", flush=True)
```

`TextSession` creates a session with `text: true`, which joins no call, transcribes nothing
and speaks nothing. `ask` streams back what the backend did on its way to an answer:

| `event.type` | what it means |
| --- | --- |
| `delta` | more of the answer, as it is written |
| `answer` | the turn is finished |
| `looked_up` | the knowledge base was searched; `query` and `documents` say what for and how much came back |
| `delegated` | work went to a skill; `skill` and `text` say which and what for |
| `settled` | that work came back, or `error` says why it did not |
| `error` | the turn failed |

## Learn More

- [The acceleration backend](../../../acceleration/README.md)
- [The stream plugin](../../plugins/stream/README.md)
- [Running the same agent on a call](../00_accelerated_example/README.md)
