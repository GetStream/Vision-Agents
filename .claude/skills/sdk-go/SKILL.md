---
name: sdk-go
description: How the Go SDK in sdks/go is built, tested and released. Read before changing sdks/go, adding a resource method, or cutting a Go SDK version.
---

# Go SDK conventions

The per-language half of [sdk](../sdk/SKILL.md) for [`sdks/go`](../../../sdks/go), module
`github.com/GetStream/Vision-Agents/sdks/go`. Go goes first (AGENTS.md "SDK changes"), so a
shared API change lands here before the other SDKs. Write the Go itself by the
[go](../go/SKILL.md) and [go-testing](../go-testing/SKILL.md) skills.

## Layout

| Package | What it is |
| --- | --- |
| `agents` | An agent spelled out in code: `agents.New`, folder sync, tools, phone, dispatch workers (`agents.NewDispatch`). What most programs start from |
| `client` | Agents addressed by name (`client.New(backend).Agent("docs")`), their sessions, responses, items, forks, guests, usage callbacks |
| `stream` | `Backend` (where the router is, who is billed), `Pipeline` (one session and its socket), `RouterError`, routers (`stream.NewClient(backend).Router(name)`), phone |
| `tools` | The function registry and the JSON Schema read off an argument struct |
| `edge` | Stream Video calls and the monitor link, through `getstream-go` |
| `acceleration` | Generated from the spec. Never edited by hand |

`stream.Backend` is the one value every package shares. A client's `Backend()` carries its
usage watcher, so an agent, worker or router built on it reports to that client's
`WithUsage` and `WithBalanceWarning`; `client.New` with no options on such a backend shares
the watcher rather than wrapping it twice.

## Go version

The `go` directive is the floor, 1.25. `go.work` and `acceleration` are on newer Go, so the
workspace compiles code 1.25 cannot: check the stdlib function you reach for exists in 1.25.
CI's `go-sdk` job builds, vets and tests `sdks/go` on Go 1.25.0 with `GOWORK=off
GOTOOLCHAIN=local`, and checks `go mod tidy -diff` and `go mod verify`. Run the same before a PR:

```bash
cd sdks/go
GOWORK=off GOFLAGS= GOTOOLCHAIN=go1.25.0 go test -race ./...
GOWORK=off go mod tidy -diff
```

No `replace` directive, ever: a consumer gets the module alone, from the proxy.

## The generated client

`go generate ./...` in `sdks/go` runs oapi-codegen on `../../acceleration/api/openapi.yaml` with
`api/oapi-codegen.yaml`, which leaves out the `Staff` operations and the four sockets
(`stream/socket.go` holds those by hand). The output is committed; CI's `go` job regenerates
it and fails on a diff. The spec is only on `accelerate`, so generation runs there.

Models public, operations not:

- Generated types (`acceleration.Session`, `AgentConfig`, `SttOptions`, enums) are the SDK's
  models and may appear in signatures. Do not wrap one for its own sake.
- Generated operations (`*WithResponse`) are never the API: README, examples and docs call a
  method of `client`, `agents` or `stream`. A new endpoint is finished when it has one.
- `stream.Backend.Client()` and `agents.Agent.Client()` hand out the generated client for an
  operation with no method yet. Their doc comments say it carries no compatibility promise;
  keep that true and add no third accessor.
- A parameter never takes `*acceleration.ClientWithResponses`. Take a `stream.Backend`.

Known gaps, generated only (docs on getstream.io call some of them raw today): simulations and
their runs, voices, knowledge documents and URLs beyond ingest and add, scheduled tasks,
campaigns, channels, connectors and connections, SIP trunks, 10DLC and opt-outs, phone
transfer, digits and release, account, credits, spend limits, usage events, teams, prices,
stats, policies, provider keys, runs, OCR and classify. Folder sync does not read
`resume_history:` from agent.yaml yet, so one that names it is refused.

## Errors, retries and idempotency

- Every refusal is a `*stream.RouterError`, built by `stream.NewRouterError` from the answer.
  Transport failures are wrapped with `%w`, so `errors.Is(err, context.Canceled)` holds.
- The SDK retries no HTTP request: it cannot know which of the caller's are safe to send twice.
  `RouterError.RetryAfter` says how long the router asked for.
- `Responses.Create` sends a fresh `request_id` per text question, so the router answers a
  resend once. A tool call is keyed by its id (`tools.Registry` receipts), so a call the router
  sends again is joined or answered from its first result.
- A text session's socket is the one thing reconnected: doubling waits from one second to 30,
  for five minutes (`Pipeline.firstRetry`, `resumeFor`), giving up on a 4xx other than 429.

## Tests

```bash
cd sdks/go
go test -race ./...
go vet ./... && go vet -tags e2e .
```

No mocks: each package has its own stand-in router, an `httptest.Server` with a real gorilla
upgrader (`router` in `client/client_test.go` and `stream/accelerated_test.go`, `restartable`
and `reopened` in `stream/resume_test.go`, `pool` in `stream/dispatch_test.go`, `worked` in
`agents/dispatch_test.go`, `backend` in `agents/agent_test.go`, `moneyRouter` in
`client/usage_test.go`, `vendors` in `stream/phone_test.go`). Reuse the package's own before
writing another. `readme_test.go` holds the README's quickstart to `examples/quickstart`.

`e2e_test.go` (tag `e2e`) drives a real router and real models, so it costs money and never
runs in CI. Without `STREAM_ACCELERATION_URL` it skips; with `E2E_REQUIRED=1` every skip fails
instead. Its package comment names the release smoke set; run it against a router that
searches, holds knowledge and bills:

```bash
E2E_REQUIRED=1 go test -tags e2e -count=1 -v -run "$SMOKE" .
```

## README

The README renders on pkg.go.dev too, where only links inside the module resolve: a link
outside `sdks/go` is an absolute `https://github.com/GetStream/Vision-Agents/tree/accelerate/...`.
Every identifier in a code block exists; the quickstart is the example file verbatim.

## Releasing

`sdks/go` is developed on `accelerate` and released from `main`, the default branch, so
`go get ...@latest`, `@main` and pkg.go.dev see it. `main` holds a copy of `sdks/go` from one
`accelerate` commit and is never edited there. A version is a tag `sdks/go/vX.Y.Z` on the
`main` merge commit. While 0.x, a minor may break a signature; a patch only fixes.

1. Land the change on `accelerate`, with the `CHANGELOG.md` entry.
2. Run the smoke set with `E2E_REQUIRED=1`, as above, and keep its output.
3. Copy onto a branch off `main`, and check the copy is exact:

   ```bash
   git fetch origin
   git switch -c ai-xxxx-go-sdk-vX.Y.Z origin/main
   SRC=$(git rev-parse origin/accelerate)
   git checkout "$SRC" -- sdks/go
   git diff --stat "$SRC" -- sdks/go   # prints nothing
   ```

   Commit, and open a PR to `main` titled `[AI-xxxx] Go SDK vX.Y.Z`, with the source SHA and the
   smoke run in the body. `main`'s `go` CI job checks the copy on Go 1.25 alone.
4. Once it is merged, tag the merge commit. Never create a GitHub Release for it: `release.yml`
   publishes the Python packages on any published release.

   ```bash
   git fetch origin
   git tag sdks/go/vX.Y.Z origin/main
   git push origin sdks/go/vX.Y.Z
   ```

5. `release-sdk-go.yml` runs on the tag: it checks the tagged copy and builds the quickstart in
   an empty module through proxy.golang.org. By hand, with `GOPRIVATE` cleared:

   ```bash
   GOPRIVATE= GONOSUMDB= GOFLAGS=-mod=mod GOPROXY=https://proxy.golang.org \
     go list -m -versions github.com/GetStream/Vision-Agents/sdks/go
   ```

A version found broken is never moved or deleted: the proxy and the checksum database keep it.
Add `retract vX.Y.Z // why` to `sdks/go/go.mod` on `accelerate`, and release the next patch the
same way; `@latest` then skips the retracted one.

## Reject in review

- A `*WithResponse` call in the README, an example or a docs page where a method exists.
- A parameter of a generated client type, or a new accessor for the generated client.
- A retry of an HTTP request without an owner, a bound and a named threat (AGENTS.md).
- A stdlib or language feature newer than the `go` directive.
- An edit to `sdks/go` on `main`.
