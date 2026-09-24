---
name: sdk-rust
description: How to build and extend the Rust server-side SDK in sdks/rust. Read this before changing the client, the generator, sockets, the agent or dispatch, or before adding a dependency.
---

# Rust SDK conventions

The per-language half of [sdk](../sdk/SKILL.md), server side only. It records the decisions
[`sdks/rust`](../../../sdks/rust) already follows, so a change lands consistently rather than
re-litigating them.

Edition 2024, MSRV 1.88 (async closures and let chains are used). One crate, `vision-agents`,
plus the `generate` binary in the same workspace, which is not published.

## Dependencies

tokio, reqwest 0.13 (`rustls`, `json`, `query`), tokio-tungstenite 0.30
(`rustls-tls-webpki-roots`), serde, thiserror 2, tokio-util (`CancellationToken`), yaml-rust2
for `agent.yaml`, md-5 for the folder fingerprint, hmac + sha2 + base64 for JWTs. axum and
tempfile are dev dependencies only.

- **rustls everywhere, no native-tls.** No OpenSSL on the build host, and the same TLS in the
  HTTP client and the socket.
- **JWTs are signed by hand** (`backend::sign`), not with `jsonwebtoken`. HS256 is an HMAC over
  two base64 segments; the crate would bring `ring` or `aws-lc` for the one algorithm used.
- **No Stream SDK.** The official `getstream` crate (0.1.0-preview.2) hard-depends on libvpx,
  openh264, opus and a WebRTC stack. The agent needs one endpoint,
  `POST {base}/api/v2/video/call/{type}/{id}?api_key=` with a server JWT and
  `Stream-Auth-Type: jwt`, plus user tokens it can already sign. That is `stream.rs`. Revisit
  when a Stream server crate exists without media.
- A new dependency is a design question. Ask first.

## Generation: typify for types, our own generator for operations

`cargo run -p generate` reads `acceleration/api/openapi.yaml` and writes `src/types.rs`
(typify 0.8) and `src/operations.rs` (one `impl Client` method per operation, built with
syn + quote + prettyplease, then rustfmt). Both are committed; `-- --check` fails when they
are out of date with the spec. `lib.rs` marks them `#[rustfmt::skip]` so `cargo fmt` does not
fight the generator.

Why not the alternatives, researched when the SDK was written:

- **progenitor** generates a whole client, and its runtime would own auth. Auth here depends
  on the credential (customer id, server JWT, user token, the proxy spelling) and the same
  headers go on the socket upgrade, which progenitor does not generate. Its
  `progenitor_client::Error` would leak into every signature, and responses with more than
  one success shape come out awkwardly.
- **openapi-generator** (Java) mishandles `allOf` and emits a crate of its own with its own
  reqwest version.
- **typify alone for types** keeps the request layer ours: one `Client::send` that signs,
  sends, decodes an empty body as `null` and maps a refusal to `Error::Router`. The generated
  methods are one-line calls into it.

The generator rewrites the spec before typify sees it: OpenAPI `nullable` becomes a JSON
Schema type union, and **schema `default`s are dropped**, so no default is ever sent that the
caller did not choose (nil means omit, as in every SDK). Structs get `Default` so a request is
`..Default::default()`.

Trade-off to know: typify's enums are strict. A value the spec does not list fails decoding
with `Error::Decode`. That is right for request types, and means a router that ships a new
enum value before the spec says so breaks this SDK. Fix the spec and regenerate; do not add
catch-all variants by hand. Never hand-edit the generated files.

## Errors

One `thiserror` enum, `vision_agents::Error`: `Router{status, operation, message}`,
`Transport`, `Decode`, `Socket`, `Failed{operation, message}` (a job or socket that reported
failure), `Closed`, `Configuration` (refused before any request), `Folder`, `Io`.
`Error::status()` answers the HTTP status when there is one. A 503 from `/health` is an error
even though it carries a `HealthStatus`: a degraded router is not an answer to act on.

## Sockets

Hand-written in `socket.rs`, because OpenAPI stops at the upgrade. `Frame` is a loosely typed
JSON object (`kind`, `text`, `flag`, `number`, `nested`), so a frame this SDK has not heard
of reaches the caller rather than being dropped. Non-JSON text frames are skipped.

- Credentials go on the upgrade as headers, from the same code as HTTP. A refused upgrade
  becomes `Error::Router` with the router's message.
- **No reconnection.** `respond` and `tool_result` are not idempotent and there is no resume.
- Modality sockets send `target` at the top level of `start` as well as in the options block:
  the router refuses a start frame whose top level names neither a target nor a config.

## Sessions answer tool calls themselves

`Session::open` spawns a watcher that reads the socket, answers `tool_call` (up to 16 at once,
each its own task) and handles `tool_cancel`, whether or not anybody calls `next_event`.
Events are buffered 256 deep, dropping the oldest. A tool's `Err` goes back as the tool's
error. Output is always sent as a string.

Drop closes the session (cancels the watcher, sends close). `within(async |session| ...)` is
the scope in place of `async with`: it closes and waits up to 5s whatever the closure
returned. `close()` is idempotent.

## Agent, folder, dispatch

- `Agent::new(config)` runs a stored config; `from_folder` loads a directory; `named` is an
  agent with no config. Builders set what the request carries; code wins over config.
- The request is built by merging JSON (`overlay`), so an unset field is absent, not null.
- `folder.rs` is a port of the Go SDK: the same lookup (`examples/*/name`, `agents/name`,
  `name`, walking up), the same strict `agent.yaml` keys, and the same MD5 fingerprint, pinned
  by a test against the Go value. Changing the hash means changing it in every SDK at once.
- `sync` on a folder is one `POST /v1/agents/sync` with the fingerprint, then `.agent_sync`.
- `Dispatch` runs handlers in a `JoinSet` and drains it on the way out; a panicking handler is
  reported as rejected. `load` and `ping` every 15s. `get_or_create_agent` keeps one session
  per channel and closes them all when `run` ends.
- `Knowledge::add_url` and `Router::transcribe`/`record` poll until the job settles.
- Speech-to-speech is not wrapped; it is reachable through `Client::socket`.

## Tests

```bash
docker run --rm -v "$REPO":/repo -v va-rust-cargo:/usr/local/cargo/registry \
  -v va-rust-rustup:/usr/local/rustup -v va-rust-target:/repo/sdks/rust/target \
  -w /repo/sdks/rust rust:1 cargo test
```

Run clippy with `--workspace --all-targets -- -D warnings`, `fmt --all --check` and
`run -q -p generate -- --check` the same way. Named volumes hold the registry and target, so
nothing lands on the host.

- **A real in-process axum server** (`tests/support`), never a mock. It records every request
  (`Seen`), answers queued routes, and hands websocket upgrades to the test as `Accepted`, so
  a test scripts frames and asserts on what the SDK actually put on the wire.
- Assert on outputs, state and what reached the server. Never that a method was called.
- `StreamApp::with_base_url` points call creation at the test server; tokens are checked by
  verifying the HS256 signature.
- Test agents use names no real directory has, or `Agent::new` will find and sync one.
- `tests/live.rs` runs only when `VISION_AGENTS_URL` is set (optional
  `VISION_AGENTS_CUSTOMER_ID`, default `examples`). Add `--add-host=host.docker.internal:host-gateway`,
  `--env-file .env` and `-e STREAM_ACCELERATION_URL=` so the host's value does not win.

## Reviewing a change

Reject it if it:

- adds native-tls, a Stream SDK, a JWT crate, or any dependency without asking;
- hand-edits `src/types.rs` or `src/operations.rs`, or wraps a single endpoint by hand;
- sends a field the caller did not set, or copies a schema default into a request;
- handles a tool call anywhere a caller must be reading events for it to run;
- treats an unknown frame as fatal, or adds reconnection;
- replaces the axum server with a mock, or asserts on calls rather than on the wire.
