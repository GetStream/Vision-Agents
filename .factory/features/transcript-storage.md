# Transcript storage

[Sprint 4](../sprint4.md), "Message storage".

## Asked for

Store the chat or transcription messages in a Stream Chat channel named after the agent id.

## What exists

[internal/chatlog](../../acceleration/internal/chatlog) writes every relevant transcript the
flow controller commits and
every reply into `messaging:{agentID}`. With `STREAM_API_KEY` and `STREAM_API_SECRET` set,
`cmd/agent` starts one and feeds it the agent's event stream.

Each message is authored by whoever said it — the caller under their own user id, the agent under
its own — so the channel reads as a conversation between two people rather than a log with a
speaker prefix. A participant is identified by user id rather than the per-call session id, which
would give them a new identity on every call. Any Stream Chat client can already open the result,
which is the reason for a channel rather than a table: a voice call otherwise leaves nothing
behind, and nothing new has to be built to read what was said.

## Two things about how it is wired

- **It consumes the event stream, not the conversation loop.** `chatlog.Record` is fed from the
  same `Events()` channel that `cmd/agent` prints from, so the conversation does not know it
  exists and a Chat outage cannot slow a reply down.
- **Failing to start is not fatal.** Without Stream credentials `cmd/agent` warns and carries
  on. The call still happens; it just is not kept.

## Not done

Nothing outstanding.
