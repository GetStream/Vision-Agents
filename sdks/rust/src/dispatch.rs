use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as SyncMutex};
use std::time::Duration;

use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::client::Client;
use crate::error::{Error, Result};
use crate::responses::{AgentResponse, Responses};
use crate::session::Session;
use crate::sessions::AgentRef;
use crate::socket::{Frame, Incoming, SocketSender};
use crate::tools::Tools;
use crate::types;

/// How many calls a worker takes at once when it does not say.
const DEFAULT_CAPACITY: u32 = 4;

/// How often the worker tells the router how it is doing.
const REPORT_EVERY: Duration = Duration::from_secs(15);

/// A call that arrived, as the router hands it to a worker.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InboundCall {
    /// The Stream call the caller is already in, which is the one to join.
    pub call_id: String,
    pub call_type: String,
    /// The number they rang, which is what the agent acts from and can transfer on.
    pub called_number: String,
    /// The number they rang from, where the vendor passed it on.
    pub caller_number: String,
    /// Whatever was put on the Stream call, for a deployment that routes on its own fields.
    pub custom: BTreeMap<String, String>,
    /// When the call arrived, as the router wrote it.
    pub at: String,
}

impl InboundCall {
    fn of(frame: &Frame) -> Self {
        InboundCall {
            call_id: frame.text("call_id").into(),
            call_type: or(frame.text("call_type"), "default"),
            called_number: frame.text("called_number").into(),
            caller_number: frame.text("caller_number").into(),
            custom: frame
                .nested("custom")
                .map(|custom| {
                    custom
                        .iter()
                        .map(|(key, value)| {
                            let value = value
                                .as_str()
                                .map_or_else(|| value.to_string(), str::to_string);
                            (key.clone(), value)
                        })
                        .collect()
                })
                .unwrap_or_default(),
            at: frame.text("at").into(),
        }
    }
}

/// A message written to an agent that is not running, or to a running session whose agent
/// leaves text to dispatch.
///
/// Otherwise one written to an agent that is already running never arrives here: the router
/// answers it from that session, because that agent knows what has been said so far.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InboundMessage {
    /// The channel it was written in, which is the conversation to answer in.
    pub channel_id: String,
    pub channel_type: String,
    /// Who to answer as, which names the channel replies are written into.
    pub agent_id: String,
    /// The stored agent config the router matched, if it matched one.
    pub config_id: String,
    /// The running session it was written to, when its agent leaves text to dispatch.
    /// Nothing has answered it: [`Dispatch::answer`] has the model do so.
    pub session_id: String,
    /// The durable command it was sent as, which the answer is created on. May be empty.
    pub command_id: String,
    pub text: String,
    pub message_id: String,
    pub user_id: String,
    pub user_name: String,
    pub at: String,
}

impl InboundMessage {
    fn of(frame: &Frame) -> Self {
        InboundMessage {
            channel_id: frame.text("channel_id").into(),
            channel_type: or(frame.text("channel_type"), "agent"),
            agent_id: frame.text("agent_id").into(),
            config_id: frame.text("config_id").into(),
            session_id: frame.text("session_id").into(),
            command_id: frame.text("command_id").into(),
            text: frame.text("text").into(),
            message_id: frame.text("message_id").into(),
            user_id: frame.text("user_id").into(),
            user_name: frame.text("user_name").into(),
            at: frame.text("at").into(),
        }
    }
}

type CallHandler = Arc<dyn Fn(InboundCall) -> BoxFuture<'static, Result<()>> + Send + Sync>;
type MessageHandler = Arc<dyn Fn(InboundMessage) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// One set of functions this worker runs for every session under an agent id.
#[derive(Clone)]
struct Hosting {
    agent_id: String,
    tools: Tools,
    tool_timeout: Option<Duration>,
}

impl Hosting {
    fn declaration(&self) -> Value {
        let tools: Vec<Value> = self
            .tools
            .declared()
            .into_iter()
            .map(|tool| json!({"name": tool.name, "description": tool.description, "parameters": tool.parameters}))
            .collect();
        let timeout_ms = self
            .tool_timeout
            .map_or(0, |timeout| timeout.as_millis() as u64);
        json!({"type": "host_tools", "agent_id": self.agent_id, "tools": tools, "timeout_ms": timeout_ms})
    }

    fn runs(&self, name: &str) -> bool {
        self.tools.declared().iter().any(|tool| tool.name == name)
    }
}

/// Waits for inbound calls and messages, and runs a handler for each one.
///
/// Neither arrives here first: a caller reached a Stream call over SIP, or somebody wrote in
/// a channel, and the router found out by webhook. So this connects out and waits, and the
/// router pushes work down the connection; nothing here has to be publicly reachable.
/// Several workers can wait at once and share the work.
///
/// Server side only: a worker is offered other people's callers.
///
/// ```no_run
/// # async fn example(client: vision_agents::Client) -> vision_agents::Result<()> {
/// use vision_agents::{Agent, Dispatch};
///
/// let dispatch = Dispatch::new(client);
/// dispatch.wait_for_call(|call| async move {
///     let session = Agent::new("support").answer(&call).await?;
///     session.wait().await;
///     Ok(())
/// });
/// dispatch.run().await
/// # }
/// ```
#[derive(Clone)]
pub struct Dispatch {
    inner: Arc<Inner>,
}

struct Inner {
    client: Client,
    capacity: u32,
    on_call: SyncMutex<Option<CallHandler>>,
    on_message: SyncMutex<Option<MessageHandler>>,
    hosted: SyncMutex<Vec<Hosting>>,
    /// Which session is answering which channel. A channel is one conversation, so the
    /// session that answered the last message on it should answer the next.
    answering: Mutex<HashMap<String, Arc<Session>>>,
    /// The calls and messages being handled, which is what the router counts against
    /// capacity. Hosted tool calls are not among them.
    handling: Arc<AtomicUsize>,
    worker_id: SyncMutex<String>,
    stop: CancellationToken,
}

impl Dispatch {
    pub fn new(client: Client) -> Self {
        Dispatch::with_capacity(client, DEFAULT_CAPACITY)
    }

    /// A worker holding up to `capacity` calls at once. The router passes over a worker
    /// that is full rather than queueing behind it, so this is a promise about what this
    /// process can actually answer.
    pub fn with_capacity(client: Client, capacity: u32) -> Self {
        Dispatch {
            inner: Arc::new(Inner {
                client,
                capacity: capacity.max(1),
                on_call: SyncMutex::new(None),
                on_message: SyncMutex::new(None),
                hosted: SyncMutex::new(Vec::new()),
                answering: Mutex::new(HashMap::new()),
                handling: Arc::new(AtomicUsize::new(0)),
                worker_id: SyncMutex::new(String::new()),
                stop: CancellationToken::new(),
            }),
        }
    }

    /// What the router calls this connection, once it has said.
    pub fn worker_id(&self) -> String {
        self.inner.worker_id.lock().expect("worker id").clone()
    }

    /// Registers what to do with an arriving call.
    ///
    /// The handler runs as its own task, so one long call does not stop the next from being
    /// answered. An error it returns is reported to the router with the call's `done`.
    pub fn wait_for_call<F, Fut>(&self, handler: F) -> &Self
    where
        F: Fn(InboundCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let handler: CallHandler = Arc::new(move |call| handler(call).boxed());
        *self.inner.on_call.lock().expect("handler") = Some(handler);
        self
    }

    /// Registers what to do with a message written to an agent that is not running, or to a
    /// session whose agent leaves text to dispatch. It runs as its own task, the way a call's
    /// handler does.
    pub fn wait_for_message<F, Fut>(&self, handler: F) -> &Self
    where
        F: Fn(InboundMessage) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let handler: MessageHandler = Arc::new(move |message| handler(message).boxed());
        *self.inner.on_message.lock().expect("handler") = Some(handler);
        self
    }

    /// Runs the agent's [`tools`](AgentRef::tools) for every session opened under its name,
    /// whoever opened it.
    ///
    /// A session's own functions run in the process that opened it, which is no use to a
    /// conversation opened from a browser. Hosting is the other direction: the router offers
    /// these functions to each session naming the agent and sends every call to a worker
    /// hosting them. `tool_timeout` is how long the router waits for one tool call to be
    /// answered before telling the model it failed, not how long the worker runs; `None`
    /// takes the router's default of two minutes. Call before [`Dispatch::run`].
    ///
    /// ```no_run
    /// # async fn example(client: vision_agents::Client) -> vision_agents::Result<()> {
    /// use vision_agents::Dispatch;
    ///
    /// let agent = client.agent("my-agent");
    /// agent.tools.register(
    ///     "get_weather",
    ///     "The weather in a city",
    ///     serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
    ///     async |arguments| Ok::<_, String>(format!("sunny in {}", arguments["city"])),
    /// );
    /// let dispatch = Dispatch::new(client);
    /// dispatch.host(&agent, None);
    /// dispatch.run().await
    /// # }
    /// ```
    pub fn host(&self, agent: &AgentRef, tool_timeout: Option<Duration>) -> &Self {
        self.inner.hosted.lock().expect("hosted").push(Hosting {
            agent_id: agent.name.clone(),
            tools: agent.tools.clone(),
            tool_timeout,
        });
        self
    }

    /// The session answering on this message's channel, started if none is.
    ///
    /// The second message on a channel goes to the session that answered the first, which
    /// is still open and knows what has been said; only a channel nothing is answering calls
    /// `create`. Sessions are kept until this worker stops waiting. A message that already
    /// has a session is refused: answer it with [`Dispatch::answer`].
    pub async fn get_or_create_agent<F, Fut>(
        &self,
        message: &InboundMessage,
        create: F,
    ) -> Result<Arc<Session>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Agent>>,
    {
        if !message.session_id.is_empty() {
            return Err(Error::configuration(
                "a session is already holding this conversation; answer it there with answer",
            ));
        }
        let mut answering = self.inner.answering.lock().await;
        if let Some(open) = answering.get(&message.channel_id)
            && open.live()
        {
            return Ok(open.clone());
        }
        let session = Arc::new(create().await?.reply(message).await?);
        answering.insert(message.channel_id.clone(), session.clone());
        Ok(session)
    }

    /// Has the model answer a message written to a session whose agent leaves text to
    /// dispatch.
    ///
    /// The response is created with this worker's own credential acting for whoever wrote
    /// the message, so it reaches their conversation and goes to the model rather than back
    /// to a worker. It carries the message's command, so the answer lands on it.
    pub async fn answer(&self, message: &InboundMessage) -> Result<AgentResponse> {
        if message.session_id.is_empty() {
            return Err(Error::configuration(
                "no session is holding this message; open one with get_or_create_agent",
            ));
        }
        Responses::new(
            self.inner.client.acting_for(&message.user_id),
            &message.session_id,
        )
        .create_with(&types::CreateResponseRequest {
            text: message.text.clone(),
            command_id: (!message.command_id.is_empty()).then(|| message.command_id.clone()),
            ..Default::default()
        })
        .await
    }

    /// Waits for calls, messages and hosted tool calls until the router closes the
    /// connection, refuses the tools this worker hosts, or [`Dispatch::stop`] is called.
    ///
    /// Work still being handled is waited for on the way out, because dropping a call would
    /// hang up on whoever is talking.
    pub async fn run(&self) -> Result<()> {
        let (on_call, on_message, hosted) = (
            self.inner.on_call.lock().expect("handler").clone(),
            self.inner.on_message.lock().expect("handler").clone(),
            self.inner.hosted.lock().expect("hosted").clone(),
        );
        if on_call.is_none() && on_message.is_none() && hosted.is_empty() {
            return Err(Error::configuration(
                "register a handler with wait_for_call or wait_for_message, or host tools, first",
            ));
        }

        // Always said, even when it is nothing: a worker that only hosts tools answers neither.
        let handles: Vec<&str> = [
            (on_call.is_some(), "call"),
            (on_message.is_some(), "message"),
        ]
        .into_iter()
        .filter_map(|(handled, kind)| handled.then_some(kind))
        .collect();
        let socket = self
            .inner
            .client
            .socket(&format!(
                "/v1/dispatch?capacity={}&active={}&handles={}",
                self.inner.capacity,
                self.inner.handling.load(Ordering::SeqCst),
                handles.join(",")
            ))
            .await?;
        let (sender, mut receiver) = (socket.sender, socket.receiver);
        let mut running: JoinSet<()> = JoinSet::new();
        let started = Instant::now();
        let mut latency_ms = 0.0;
        let mut report = tokio::time::interval_at(Instant::now() + REPORT_EVERY, REPORT_EVERY);
        let mut failure = None;

        loop {
            while running.try_join_next().is_some() {}
            let incoming = tokio::select! {
                _ = self.inner.stop.cancelled() => break,
                _ = report.tick() => {
                    tell(&sender, json!({"type": "load", "active_agents": running.len(), "latency_ms": latency_ms})).await;
                    // Timed from this side, because this is the side audio crosses.
                    tell(&sender, json!({"type": "ping", "at": started.elapsed().as_secs_f64()})).await;
                    continue;
                }
                incoming = receiver.next() => incoming,
            };
            let frame = match incoming {
                Some(Ok(Incoming::Frame(frame))) => frame,
                Some(Ok(Incoming::Audio(_))) => continue,
                Some(Err(_)) | None => break,
            };

            match frame.kind() {
                "call" => {
                    let work_id = frame.text("work_id").to_string();
                    match on_call.clone() {
                        Some(handler) => {
                            let call = InboundCall::of(&frame);
                            self.work(&mut running, &sender, work_id, move || handler(call));
                        }
                        None => {
                            tell(
                                &sender,
                                done(&work_id, Some("this worker answers no calls".into())),
                            )
                            .await
                        }
                    }
                }
                "message" => {
                    let work_id = frame.text("work_id").to_string();
                    match on_message.clone() {
                        Some(handler) => {
                            let message = InboundMessage::of(&frame);
                            self.work(&mut running, &sender, work_id, move || handler(message));
                        }
                        None => {
                            tell(
                                &sender,
                                done(&work_id, Some("this worker answers no messages".into())),
                            )
                            .await
                        }
                    }
                }
                "ready" => {
                    *self.inner.worker_id.lock().expect("worker id") =
                        frame.text("worker_id").into();
                    for offer in &hosted {
                        tell(&sender, offer.declaration()).await;
                    }
                }
                // Not awaited inline: a tool can take a minute, and this socket also delivers the next.
                "tool_call" => {
                    let (id, name) = (frame.text("id").to_string(), frame.text("name").to_string());
                    let Some(offer) = hosted.iter().find(|offer| offer.runs(&name)) else {
                        tell(&sender, json!({"type": "tool_result", "id": id, "error": format!("this worker does not run {name}")})).await;
                        continue;
                    };
                    let (tools, sender) = (offer.tools.clone(), sender.clone());
                    let arguments = frame.text("arguments").to_string();
                    running.spawn(async move {
                        let outcome =
                            tokio::spawn(async move { tools.call(&name, &arguments).await }).await;
                        let result = match outcome {
                            Ok(Ok(output)) => json!({"type": "tool_result", "id": id, "output": output}),
                            Ok(Err(error)) => json!({"type": "tool_result", "id": id, "error": error}),
                            Err(_) => json!({"type": "tool_result", "id": id, "error": "the tool panicked"}),
                        };
                        tell(&sender, result).await;
                    });
                }
                "hosting_refused" => {
                    // A worker whose tools were refused is one nobody will call.
                    failure = Some(Error::Failed {
                        operation: "dispatch".into(),
                        message: format!(
                            "the router refused to host tools for agent {}: {}",
                            frame.text("agent_id"),
                            frame.text("reason")
                        ),
                    });
                    break;
                }
                "pong" => {
                    latency_ms = (started.elapsed().as_secs_f64() - frame.number("at")) * 1000.0
                }
                _ => {}
            }
        }

        while running.join_next().await.is_some() {}
        let answering: Vec<Arc<Session>> = self
            .inner
            .answering
            .lock()
            .await
            .drain()
            .map(|(_, session)| session)
            .collect();
        for session in answering {
            session.close().await;
        }
        sender.close().await;
        failure.map_or(Ok(()), Err)
    }

    /// Stops waiting. Work already being handled is still waited for by `run`.
    pub fn stop(&self) {
        self.inner.stop.cancel();
    }

    /// Runs one call or message as its own task, then tells the router it is `done`, which is
    /// what gives this worker its room back.
    fn work<F>(&self, running: &mut JoinSet<()>, sender: &SocketSender, work_id: String, start: F)
    where
        F: FnOnce() -> BoxFuture<'static, Result<()>> + Send + 'static,
    {
        let (sender, handling) = (sender.clone(), self.inner.handling.clone());
        handling.fetch_add(1, Ordering::SeqCst);
        running.spawn(async move {
            // Spawned again so a handler that panics is reported like one that failed.
            let error = match tokio::spawn(async move { start().await }).await {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error.to_string()),
                Err(_) => Some("the handler panicked".to_string()),
            };
            handling.fetch_sub(1, Ordering::SeqCst);
            tell(&sender, done(&work_id, error)).await;
        });
    }
}

/// The frame that ends one piece of work, with what went wrong if anything did.
fn done(work_id: &str, error: Option<String>) -> Value {
    let mut done = json!({"type": "done", "work_id": work_id});
    if let Some(error) = error {
        done["error"] = error.into();
    }
    done
}

/// Sends a frame if the socket is still there. None of these is something a call depends on.
async fn tell(sender: &SocketSender, frame: Value) {
    if sender.open() {
        let _ = sender.send(&frame).await;
    }
}

fn or(value: &str, fallback: &str) -> String {
    if value.is_empty() {
        fallback.into()
    } else {
        value.into()
    }
}
