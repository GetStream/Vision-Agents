# The Go SDK

[Sprint 10](../sprint10.md).

## Asked for

`agents-core-go`, generated in part from the acceleration OpenAPI. Every agent can be a
directory holding its skills, its knowledge and its instructions. A text example and a voice
example, both shaped like the Python ones, plus buying a number, waiting for a call on it and
calling somebody.

## What exists

[agents-core-go](../../agents-core-go) is the Go counterpart of
[plugins/stream](../../plugins/stream), not of `agents-core`. It creates sessions on the
backend, holds the event socket, runs local functions and configures agents. The pipeline
itself runs in the router, the same as it does for Python.

| Package                                            | What it holds                                    |
| -------------------------------------------------- | ------------------------------------------------ |
| [agents](../../agents-core-go/agents)              | `Agent`, `Session`, `Harness`, the folder loader, `Sync` |
| [stream](../../agents-core-go/stream)              | `Accelerated`, the backend client, phone, the hand-written socket |
| [edge](../../agents-core-go/edge)                  | Creating and joining a Stream call, and `MonitorURL` |
| [tools](../../agents-core-go/tools)                | Function registry, JSON Schema from struct tags   |
| [acceleration](../../agents-core-go/acceleration)  | The generated REST client                         |

```go
llm := stream.Accelerated(stream.Config{
    STT: "deepgram/flux-general-en", TTS: "cartesia/sonic-preview", LLM: "llm-fast",
})
agent, err := agents.New(agents.Options{Name: "jean", Dir: "agents/jean", LLM: llm})
session, err := agent.Join(ctx, edge.Call{})
```

`Chat` for text, `Join` for a Stream call, `WaitForCall` and `StartCall` for a phone. All
four return a `Session` with the same events, so what a caller reads does not change with how
the conversation arrived.

## An agent is a directory

`agents/jean/` holds `instructions.md`, `skills/*.md` and `knowledge/`. A skill is a markdown
file whose frontmatter carries its name, its one-line description and its deadline, and whose
body is the prompt only the subagent sees — the same three things
[skills.yaml](../../acceleration/internal/harness/skills.yaml) declares, written where
somebody editing them would look.

Every part is optional, and code wins over disk: `Options` set in Go are left alone and only
the empty ones are filled from the folder. `Sync` pushes the lot to the backend — the config,
the skills, and the knowledge under a namespace named after the folder — so the directory is
the source and the backend is a copy of it.

## Generated the same way the Python client is

`go generate ./...` runs `oapi-codegen` against `acceleration/api/openapi.yaml`, the same file
the server and the Python client come from, and the output is committed so nothing is
generated at install. The two WebSockets are excluded, as they are everywhere else, and
`stream/socket.go` is the hand-written half.

## Not done

- **No dispatch client.** Inbound in Go is `WaitForCall`, which attaches a number and waits;
  the [dispatch](dispatch.md) socket has no Go worker.
- **No per-modality client.** `stream.STT`, `TTS` and `LLM` exist in Python only, so a Go
  pipeline is the whole pipeline or none of it.
- **No wrappers for voices or campaigns.** Both are reachable through the generated client,
  which is enough to use them and not enough to be pleasant.
- **`open_monitoring` is a link.** `Session.MonitorURL()` returns one; nothing opens a browser.
