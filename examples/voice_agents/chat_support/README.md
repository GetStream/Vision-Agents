# Chat support (inbound messages)

Answers support questions written into an agent's Stream Chat channel, using the
handbook in this directory.

```bash
cd examples/voice_agents/chat_support
uv sync
uv run chat_support.py
```

Needs a router that Stream Chat delivers messages to (`chat.inbound_keys`, see
`acceleration/docs/stream-tenancy.md`): the router points no message hook any more.
The call hook is still pointed by hand:

```bash
cd acceleration
go run ./cmd/phone hooks -url $ROUTER_PUBLIC_URL
```

A local router behind a tunnel gets no message this way: Stream Chat delivers to the
router its shard names.

Stream refuses every change to the hooks while one of them is at a host that does
not resolve, such as a tunnel that is gone. Drop it, and the message hook an older
router wrote there, in the same run:

```bash
go run ./cmd/phone hooks -url $ROUTER_PUBLIC_URL -remove https://old-tunnel.example
```

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

So `source` is a reserved custom field on an `agent` channel. The router answers no
message that carries it, whatever its value, and logs nothing about it
(`acceleration/internal/api/messagehooks.go`, `addressed`). A client that writes a
person's message must not set `source` on it, or the message is never answered.
