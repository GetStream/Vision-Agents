# vision-agents

The Rust client for the Stream acceleration backend, for a server that runs agents.

Nothing here does inference or touches media. The backend joins the call, hears the caller,
answers and speaks. What arrives here are the events saying so, and what stays here is
function calling, because the functions are here.

```toml
[dependencies]
vision-agents = { path = "sdks/rust" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Rust 1.88 or newer, on tokio. TLS is rustls; there is no OpenSSL and no media stack.

## An agent

```rust
use serde_json::json;
use vision_agents::{Agent, Tools, types};

let tools = Tools::new();
tools.register(
    "get_weather",
    "Get the current weather for a city",
    json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    async |arguments| Ok::<_, String>(fetch_weather(arguments["city"].as_str()).await),
);

let agent = Agent::named("john")
    .instructions("You are a friendly assistant. Keep replies short.")
    .pipeline(types::CreateSessionRequest {
        llm: Some("llm-fast".into()),
        stt: Some("en-low-latency".into()),
        tts: Some("sonic_36".into()),
        ..Default::default()
    })
    .cost_tracking([("team", "support")])
    .memory_filter([("customer", "acme")])
    .tools(tools);

let session = agent.join("").await?; // an empty id names a new call
println!("{}", agent.monitor_url(&session)?);

while let Some(event) = session.next_event().await {
    println!("{} {}", event.kind, event.text);
}
```

`Agent::new("support")` runs from a stored config instead, and anything set in code wins over
it for these sessions only. `join` creates the Stream call and returns once the backend is in
it. `chat` holds the same conversation in writing, with nothing transcribed or spoken.

Tool calls are answered by the session itself, whether or not anything is reading events: the
model is mid-sentence waiting. A tool that returns `Err` is reported to the model.

A session closes when it is dropped. To close it at a point you choose and wait for it,
scope it:

```rust
agent.join("my-call").await?.within(async |session| {
    session.responses.create("greet the caller").await?;
    session.wait().await;
    Ok(())
}).await?;
```

## Who is calling

```rust
use vision_agents::{Client, ClientOptions};

// On your own server: STREAM_API_KEY and STREAM_API_SECRET.
let api = Client::from_env()?;

// A router with nothing in front of it, which is a laptop.
let api = Client::new(ClientOptions {
    url: Some("http://localhost:8080".into()),
    customer_id: Some("examples".into()),
    ..Default::default()
})?;
```

`url` falls back to `STREAM_ACCELERATION_URL`, then `http://localhost:8080`. A hosted
deployment behind Stream's proxy wants `authenticate: Some(true)`, or
`STREAM_ACCELERATION_AUTHENTICATE`. Pass the client to an agent with `.client(api)`.

Every operation in `acceleration/api/openapi.yaml` is a method on `Client`, typed from the spec:

```rust
let configs = api.list_agent_configs(&Default::default()).await?;
let guest = api.guest_user(&types::GuestUserRequest::default()).await?;
let as_guest = api.as_guest(&guest)?;
```

A refusal is `Error::Router`, carrying the status, the operation and what the router said.

## Agent dispatch

A caller reached a Stream call over SIP, or somebody wrote in a channel. The worker connects
out and the router pushes the work down the connection, so nothing you run has to be publicly
reachable.

```rust
use vision_agents::{Agent, Client, Dispatch};

let dispatch = Dispatch::with_capacity(Client::from_env()?, 4);

dispatch.wait_for_call(async |call| {
    let session = Agent::new("support").answer(&call).await?;
    session.wait().await;
    Ok(())
});

let worker = dispatch.clone();
dispatch.wait_for_message(move |message| {
    let worker = worker.clone();
    async move {
        let session = worker
            .get_or_create_agent(&message, async || Ok(Agent::new("support")))
            .await?;
        session.respond(&message.text).await
    }
});

dispatch.run().await?;
```

`capacity` is a promise about what this process can answer: the router passes over a full
worker rather than queueing behind it. `get_or_create_agent` keeps one session per channel,
because the session that answered the last message is the one that knows what was said.

Placing a call outward:

```rust
let session = agent.outbound_call("+15551234567", "+15557654321").await?;
```

## An agent written down as a directory

```
agents/jean/
  agent.yaml            required: the name and what it runs on (llm, stt, tts, tags, ...)
  instructions.md
  guardrail.md
  skills/think.md
  knowledge/pricing.md
  knowledge/urls.yaml
  .agent_sync           written by sync: the fingerprint last synced and when
```

```rust
let agent = Agent::from_folder("agents/jean")?;
agent.sync().await?;
agent.knowledge()?.add_url("https://example.com/pricing", "Pricing", "").await?;
```

`sync` stores the directory as a config in one request, and `.agent_sync` records its
fingerprint, so syncing on every startup sends nothing when nothing changed. A key
`agent.yaml` does not know is refused. `Agent::new("jean")` finds `agents/jean` or
`examples/*/jean` from the working directory up and syncs it before the first session.

## Skills and a sandbox

```rust
use vision_agents::{Harness, Skill, daytona};

let agent = Agent::new("jean").harness(Harness {
    vm: Some(daytona()),
    skills: vec![Skill::new("think", "Work through a hard question", "Reason step by step.")],
    ..Harness::standard()
});
```

## Going back, and branching off

```rust
let items = session.responses.items.all().await?;
session.responses.rewind(&items[2]).await?;
let branch = session.fork(&types::ForkSessionRequest {
    response_id: Some(items[2].response_id.clone()),
    title: Some("asked again".into()),
    ..Default::default()
}).await?;
```

`rewind` takes a response, its id, or any item of one. A conversation kept in Stream Chat
cannot be rewound; fork it at the response instead.

## One modality at a time

```rust
use vision_agents::Router;

let router = Router::new(api.clone());
let answer = router.search("what changed in v3", None).await?;

let mut voice = router.voice(types::TtsOptions {
    target: Some("en-low-latency".into()),
    ..Default::default()
}).await?;
voice.speak("hello", |audio| play(audio.pcm)).await?;
```

`transcriber` and `completions` are the speech and model sockets; `transcribe` and `record`
are the batch jobs. A socket names a target or a config (`Router::config`), or it is refused.

## Stream

There is no Stream server SDK here. The official `getstream` crate is a preview that pulls in
a WebRTC and codec stack, so `StreamApp` does the two things an agent needs over REST: create a
call and mint a user token. It reads `STREAM_API_KEY` and `STREAM_API_SECRET`.

## Working on this crate

Everything runs in the official Rust image; nothing is installed on the host.

```bash
alias cargo-docker='docker run --rm -v "$PWD/../..":/repo \
  -v va-rust-cargo:/usr/local/cargo/registry -v va-rust-rustup:/usr/local/rustup \
  -v va-rust-target:/repo/sdks/rust/target -w /repo/sdks/rust rust:1 cargo'

cargo-docker test                                        # against an in-process axum server
cargo-docker clippy --workspace --all-targets -- -D warnings
cargo-docker fmt --all --check
cargo-docker run -q -p generate                          # regenerate src/types.rs, src/operations.rs
cargo-docker run -q -p generate -- --check               # what CI runs
```

The generated files are committed. `acceleration/api/openapi.yaml` is the source of truth:
after editing it, regenerate every client, see [acceleration/README.md](../../acceleration/README.md).

The live suite talks to a running router and skips without one:

```bash
docker run --rm --add-host=host.docker.internal:host-gateway --env-file ../../.env \
  -e VISION_AGENTS_URL=http://host.docker.internal:8080 -e VISION_AGENTS_CUSTOMER_ID=examples \
  -e STREAM_ACCELERATION_URL= \
  -v "$PWD/../..":/repo -v va-rust-cargo:/usr/local/cargo/registry \
  -v va-rust-rustup:/usr/local/rustup -v va-rust-target:/repo/sdks/rust/target \
  -w /repo/sdks/rust rust:1 cargo test --test live
```
