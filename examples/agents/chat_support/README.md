# Chat support (inbound messages)

Answers support questions written into an agent's Stream Chat channel, using the
handbook in this directory.

```bash
cd examples/agents/chat_support
uv sync
uv run chat_support.py
```

Needs a router with message hooks pointed at it:

```bash
cd acceleration
go run ./cmd/phone hooks -url $ROUTER_PUBLIC_URL
```

That points both the call hook and the message hook, so the same router answers
a caller and a writer.

## Where a message goes

An agent writes what was said into the channel `agent:{agent_id}`, which for a
call is the call id. That channel outlives the call, so writing to it is how you
reach the agent that was on it:

- **The agent is still running.** The router answers from that session itself.
  The reply is written into the channel and is not spoken, because whoever is on
  the call did not ask the question and should not be read the answer.
- **Nothing is running.** The message is handed to this process, which answers it
  with the agent it keeps for that channel, starting one if there is none. The
  agent is given the channel, so its replies land back in the conversation.

Either way the agent's own writing carries a `source` field, so the messages it
stores are not mistaken for new questions and answered again.
