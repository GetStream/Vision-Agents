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

`url` falls back to `STREAM_ACCELERATION_URL`, then Stream's hosted router, so it is only
set for a self-hosted or local one. The hosted router is reached through Stream's proxy,
which is on by default for it; a self-hosted deployment behind the same proxy wants
`authenticate: Some(true)`, or `STREAM_ACCELERATION_AUTHENTICATE`. Pass the client to an
agent with `.client(api)`.

Every operation in `acceleration/api/openapi.yaml` is a method on `Client`, typed from the spec:

```rust
let configs = api.list_agent_configs(&Default::default()).await?;
let guest = api.guest_user(&types::GuestUserRequest::default()).await?;
let as_guest = api.as_guest(&guest)?;
```

A refusal is `Error::Router`, holding a `RouterFailure`: the status and the operation, the
router's `message`, `kind` (`not_found`, `rate_limited`, ...), `code` to branch on and
`doc_url`, and `request_id`, the `X-Request-Id` to quote to support. A body that is not the
router's envelope, such as a proxy's error page, is the message, and `kind`, `code` and
`doc_url` are left empty.

```rust
if let Err(vision_agents::Error::Router(failure)) = api.get_session("s1").await {
    eprintln!("{}: {} (request {})", failure.code, failure.message, failure.request_id);
}
```

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
        if !message.session_id.is_empty() {
            worker.answer(&message).await?;
            return Ok(());
        }
        let session = worker
            .get_or_create_agent(&message, async || Ok(Agent::new("support")))
            .await?;
        session.responses.create(&message.text).await?;
        Ok(())
    }
});

dispatch.run().await?;
```

`capacity` is a promise about what this process can answer: the router passes over a full
worker rather than queueing behind it. The worker tells the router which kinds of work it
handles, and reports each call and message `done` when its handler returns, with the error
if it failed. `get_or_create_agent` keeps one session per channel, because the session that
answered the last message is the one that knows what was said.

An agent whose `agent.yaml` says `dispatch: {text: enabled}` hands what end users write to the
worker with the running session's `session_id` and its `command_id`. `answer` has the model
answer it on that session, with the worker's own credential acting for the user who wrote it.
`get_or_create_agent` refuses such a message.

A worker can also host an agent's functions for every session opened under its name,
including ones opened from a browser. The router offers them to each session and sends every
call here:

```rust
let agent = client.agent("my-agent");
agent.tools.register(
    "get_weather",
    "The weather in a city",
    json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    async |arguments| Ok::<_, String>(format!("sunny in {}", arguments["city"])),
);
dispatch.host(&agent, None); // None: the router waits its default two minutes for each tool call
dispatch.run().await?;
```

A worker that only hosts tools needs no handler. If the router refuses the tools, `run`
returns `Error::Failed` naming the agent and the reason.

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
  knowledge/urls.yaml   pages to read, each a url or {url, title, description, refresh_hours}
  simulations/*.yaml    each a list of {name, scenario, assertion, mode, variations, ...}
  .agent_sync           written by sync: the fingerprint last synced and when
```

```rust
let agent = Agent::from_folder("agents/jean")?;
agent.sync().await?;
agent.knowledge()?.add_url("https://example.com/pricing", "Pricing", "", Some(24)).await?;
```

`sync` stores the directory as a config in one request, and `.agent_sync` records its
fingerprint, so syncing on every startup sends nothing when nothing changed. A key
`agent.yaml` does not know is refused; besides the models it takes `speed`, `harness`,
`thinking_llm`, `sandbox` and `dispatch`. With a `simulations/` directory the config's
simulations become exactly what it declares, and an empty one deletes them; without one they
are left alone. `Agent::new("jean")` finds `agents/jean` or `examples/*/jean` from the
working directory up and syncs it before the first session.

Part of a stored config is changed without restating the rest:

```rust
api.agent("jean").update_config(&types::AgentConfigPatch {
    guardrail: Some("Never quote a price.".into()),
    ..Default::default()
}).await?;
```

## Skills and a sandbox

```rust
use vision_agents::{Harness, Skill, daytona};

let agent = Agent::new("jean").harness(Harness {
    vm: Some(daytona()),
    skills: vec![Skill::new("think", "Work through a hard question", "Reason step by step.")],
    ..Harness::standard()
});
```

The harness is part of the agent's config, never of a session: `sync` stores it, and every
session opened from the config runs it.

## Reading conversations back

```rust
use vision_agents::Query;

let sessions = &api.agent("support").sessions;
let page = sessions.query(Query { state: Some("live".into()), ..Default::default() }).await?;
let next = sessions.query(Query { cursor: page.next_cursor.clone(), ..Default::default() }).await?;
let found = sessions.search("pricing", Query::default()).await?;
```

Lists page by cursor: pass a page's `next_cursor` back, with the same filters, while
`has_more` is true. `responses.list` and `items.list` page the same way; `items.all` reads
every page.

`close` stops a conversation and keeps what it recorded and remembered. Deleting is its own
call:

```rust
session.delete_memories().await?;          // what this conversation remembered
session.delete().await?;                   // the conversation, its turns and its memories
api.memories().truncate("user-123").await?; // everything remembered about one user
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

`rewind` takes a response, its id, or any item of one. A text conversation is kept in Stream
Chat unless it is `incognito`, and one kept there cannot be rewound; fork it at the response
instead.

## Changing a session

```rust
session.update(&types::UpdateSessionRequest {
    llm: Some("llm-thinking".into()),
    thinking: Some(types::UpdateSessionRequestThinking::High),
    ..Default::default()
}).await?;

// an ended session can still be renamed
api.agent("support").sessions.update(&session_id, &types::UpdateSessionRequest {
    title: Some("Pricing".into()),
    ..Default::default()
}).await?;
```

Title, description, custom, instructions, models and voice, in one call. It applies from the
next turn; a field left `None` is left as it is.

## Simulations

```rust
let simulation = api.simulations().create(&types::SimulationRequest {
    name: "lunch".into(),
    config_id: config.id.clone(),
    scenario: "Order a turkey club, then swap it for a veggie wrap.".into(),
    assertion: "The final order is one veggie wrap.".into(),
    ..Default::default()
}).await?;
let run = api.simulations().run(&simulation.id).await?;
let run = api.simulations().runs.get(&run.id).await?; // again until it is no longer running
```

## One modality at a time

A router comes from the client, named by a stored router config (`routers/clinic/router.yaml`),
which holds the target. A call passes only what it overrides:

```rust
let router = api.router("clinic");

let mut stt = router.transcriber(Default::default()).await?;
stt.send_audio(&pcm).await?; // 16 kHz mono PCM16
while let Some(frame) = stt.next().await {
    let frame = frame?;
    if frame.kind() == "transcript" && frame.flag("final") {
        println!("{}", frame.text("text"));
    }
}

let answer = router.search("what changed in v3", None).await?;

let mut voice = router.voice(Default::default()).await?;
voice.speak("hello", |audio| play(audio.pcm)).await?;
```

`transcriber` and `completions` are the speech and model sockets; `transcribe` and `record`
are the batch jobs. Options passed to any of them (`diarize`, `keyterms`, `voice`, ...)
override that field of the config.

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
