use std::collections::{BTreeMap, HashMap};
use std::future::Future;
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
use crate::session::Session;
use crate::socket::{Frame, Incoming, SocketSender};

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

/// A message written to an agent that is not running.
///
/// One written to an agent that is already running never arrives here: the router answers
/// it from that session, because that agent knows what has been said so far.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InboundMessage {
    /// The channel it was written in, which is the conversation to answer in.
    pub channel_id: String,
    pub channel_type: String,
    /// Who to answer as, which names the channel replies are written into.
    pub agent_id: String,
    /// The stored agent config the router matched, if it matched one.
    pub config_id: String,
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
    /// Which session is answering which channel. A channel is one conversation, so the
    /// session that answered the last message on it should answer the next.
    answering: Mutex<HashMap<String, Arc<Session>>>,
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
                answering: Mutex::new(HashMap::new()),
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
    /// answered. An error it returns is reported to the router as a call nobody took.
    pub fn wait_for_call<F, Fut>(&self, handler: F) -> &Self
    where
        F: Fn(InboundCall) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let handler: CallHandler = Arc::new(move |call| handler(call).boxed());
        *self.inner.on_call.lock().expect("handler") = Some(handler);
        self
    }

    /// Registers what to do with a message written to an agent that is not running. It runs
    /// as its own task, the way a call's handler does.
    pub fn wait_for_message<F, Fut>(&self, handler: F) -> &Self
    where
        F: Fn(InboundMessage) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let handler: MessageHandler = Arc::new(move |message| handler(message).boxed());
        *self.inner.on_message.lock().expect("handler") = Some(handler);
        self
    }

    /// The session answering on this message's channel, started if none is.
    ///
    /// The second message on a channel goes to the session that answered the first, which
    /// is still open and knows what has been said; only a channel nothing is answering calls
    /// `create`. Sessions are kept until this worker stops waiting.
    pub async fn get_or_create_agent<F, Fut>(
        &self,
        message: &InboundMessage,
        create: F,
    ) -> Result<Arc<Session>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Agent>>,
    {
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

    /// Waits for calls and messages until the router closes the connection or
    /// [`Dispatch::stop`] is called.
    ///
    /// Work still being handled is waited for on the way out, because dropping a call would
    /// hang up on whoever is talking.
    pub async fn run(&self) -> Result<()> {
        let (on_call, on_message) = (
            self.inner.on_call.lock().expect("handler").clone(),
            self.inner.on_message.lock().expect("handler").clone(),
        );
        if on_call.is_none() && on_message.is_none() {
            return Err(Error::configuration(
                "register a handler with wait_for_call or wait_for_message first",
            ));
        }

        let socket = self
            .inner
            .client
            .socket(&format!("/v1/dispatch?capacity={}", self.inner.capacity))
            .await?;
        let (sender, mut receiver) = (socket.sender, socket.receiver);
        let mut running: JoinSet<()> = JoinSet::new();
        let started = Instant::now();
        let mut latency_ms = 0.0;
        let mut report = tokio::time::interval_at(Instant::now() + REPORT_EVERY, REPORT_EVERY);

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
                    let Some(handler) = on_call.clone() else {
                        continue;
                    };
                    let (call, sender) = (InboundCall::of(&frame), sender.clone());
                    running.spawn(async move {
                        let call_id = call.call_id.clone();
                        // Spawned again so a handler that panics is reported like one that failed.
                        let outcome = tokio::spawn(handler(call)).await;
                        let answer = match outcome {
                            Ok(Ok(())) => json!({"type": "accepted", "call_id": call_id}),
                            Ok(Err(error)) => json!({"type": "rejected", "call_id": call_id, "reason": error.to_string()}),
                            Err(_) => json!({"type": "rejected", "call_id": call_id, "reason": "the handler panicked"}),
                        };
                        tell(&sender, answer).await;
                    });
                }
                // Nothing is reported back for a message: there is no caller waiting on a line.
                "message" => {
                    let Some(handler) = on_message.clone() else {
                        continue;
                    };
                    let message = InboundMessage::of(&frame);
                    running.spawn(async move {
                        let _ = tokio::spawn(handler(message)).await;
                    });
                }
                "ready" => {
                    *self.inner.worker_id.lock().expect("worker id") =
                        frame.text("worker_id").into()
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
        Ok(())
    }

    /// Stops waiting. Work already being handled is still waited for by `run`.
    pub fn stop(&self) {
        self.inner.stop.cancel();
    }
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
