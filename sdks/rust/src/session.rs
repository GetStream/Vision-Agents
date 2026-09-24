use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::{Notify, watch};
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::client::{Client, segment};
use crate::error::{Error, Result};
use crate::responses::Responses;
use crate::socket::{Frame, Incoming, SocketReceiver, SocketSender};
use crate::tools::Tools;
use crate::types;

/// How many events are held for a caller that is not reading them.
///
/// Past this the oldest is dropped rather than the socket stalled. Tool calls are answered
/// by the watcher rather than by whoever reads events, so dropping an event never drops a turn.
const BUFFERED_EVENTS: usize = 256;

/// How many tool calls may run at once before the model is told to carry on without one.
const RUNNING_TOOLS: usize = 16;

/// How long `close` waits for the router to end a conversation it was asked to end.
const CLOSE_GRACE: Duration = Duration::from_secs(5);

/// Somebody on the call, as the backend reports them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Participant {
    pub id: String,
    pub user_id: String,
    pub name: String,
}

/// One thing the conversation did.
///
/// `kind` is the backend's own name for it (joined, heard, responding, response_delta,
/// responded, turn, delegated, tool_ran, error, left and the rest). The fields are filled
/// from whichever frames carry them, and `frame` is the whole thing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionEvent {
    pub kind: String,
    pub text: String,
    pub participant: Option<Participant>,
    pub interrupted: bool,
    pub pending_work: bool,
    pub error: String,
    pub frame: Frame,
}

impl SessionEvent {
    fn of(frame: Frame) -> Self {
        SessionEvent {
            kind: frame.kind().to_string(),
            text: frame.text("text").to_string(),
            participant: frame.nested("participant").map(|who| Participant {
                id: who.text("id").to_string(),
                user_id: who.text("user_id").to_string(),
                name: who.text("name").to_string(),
            }),
            interrupted: frame.flag("interrupted"),
            pending_work: frame.flag("pending_work"),
            error: frame.text("error").to_string(),
            frame,
        }
    }
}

/// How the events socket is opened.
#[derive(Debug, Clone, Copy, Default)]
pub struct WatchOptions {
    /// Also report what the caller is part way through saying, as `hearing` events.
    pub interim: bool,
    /// Report the router's own routing decisions, several times a second. Off by default.
    pub decisions: bool,
}

#[derive(Default)]
struct Shared {
    events: Mutex<VecDeque<SessionEvent>>,
    arrived: Notify,
    running: Mutex<HashMap<String, AbortHandle>>,
}

/// One conversation, held in the acceleration backend.
///
/// Nothing here does inference or touches media. The backend hears the caller, answers and
/// speaks; what arrives here are the events saying so, and what stays here is function
/// calling, because the functions are here.
///
/// A session is a guard. Dropping it tells its watcher to end the conversation, which it
/// does on its own task because a drop cannot wait; [`Session::close`] is the same thing
/// awaited, and [`Session::within`] closes it once a closure is done with it.
pub struct Session {
    /// What the router said when it created this.
    pub created: types::Session,
    /// This conversation's turns, and what each of them was made of.
    pub responses: Responses,
    client: Client,
    tools: Tools,
    sender: SocketSender,
    shared: Arc<Shared>,
    ended: watch::Receiver<bool>,
    cancel: CancellationToken,
    watcher: Mutex<Option<JoinHandle<()>>>,
}

impl Session {
    /// Creates a session and starts watching it.
    ///
    /// It returns once the backend is in the call, so a session that has opened is already
    /// listening. The tools are declared on the request so the model is offered them.
    pub async fn open(
        client: &Client,
        mut request: types::CreateSessionRequest,
        tools: Tools,
        options: WatchOptions,
    ) -> Result<Session> {
        let declared = tools.declared();
        if !declared.is_empty() {
            request.tools = Some(declared);
        }
        let created = client.create_session(&request).await?;
        Session::watching(client, created, tools, options).await
    }

    /// Starts watching a session the router has already created, such as a fork.
    pub async fn watching(
        client: &Client,
        created: types::Session,
        tools: Tools,
        options: WatchOptions,
    ) -> Result<Session> {
        let mut path = format!("/v1/agents/sessions/{}/events", segment(&created.id));
        let mut query = Vec::new();
        if options.interim {
            query.push("interim=true");
        }
        if !options.decisions {
            query.push("decisions=false");
        }
        if !query.is_empty() {
            path = format!("{path}?{}", query.join("&"));
        }

        let socket = match client.socket(&path).await {
            Ok(socket) => socket,
            Err(error) => {
                // The session is live in the backend even though nothing here can watch it,
                // so it is closed rather than left holding a call nobody is listening to.
                let _ = client.close_session(&created.id).await;
                return Err(error);
            }
        };

        let shared = Arc::new(Shared::default());
        let cancel = CancellationToken::new();
        let (ended_tx, ended) = watch::channel(false);
        let watcher = tokio::spawn(watch_socket(
            socket.receiver,
            socket.sender.clone(),
            shared.clone(),
            tools.clone(),
            cancel.clone(),
            ended_tx,
        ));

        Ok(Session {
            responses: Responses::new(client.clone(), &created.id),
            created,
            client: client.clone(),
            tools,
            sender: socket.sender,
            shared,
            ended,
            cancel,
            watcher: Mutex::new(Some(watcher)),
        })
    }

    /// The backend's id for the session.
    pub fn id(&self) -> &str {
        &self.created.id
    }

    /// The Stream Chat channel replies are written into, empty for one that keeps none.
    pub fn conversation_id(&self) -> &str {
        self.created.conversation_id.as_deref().unwrap_or("")
    }

    /// Whether the conversation is still being held.
    pub fn live(&self) -> bool {
        !*self.ended.borrow()
    }

    /// The functions this conversation runs for the model.
    pub fn tools(&self) -> &Tools {
        &self.tools
    }

    /// The next thing the conversation did, or `None` once it has ended.
    ///
    /// There is one stream: two readers would take half the events each.
    pub async fn next_event(&self) -> Option<SessionEvent> {
        loop {
            let waiting = self.shared.arrived.notified();
            if let Some(event) = self.shared.events.lock().expect("events").pop_front() {
                return Some(event);
            }
            if !self.live() {
                return None;
            }
            waiting.await;
        }
    }

    /// Reads events until one of `kind` arrives, and returns it. Events before it are
    /// consumed. `None` means the conversation ended first.
    pub async fn wait_for_event(&self, kind: &str) -> Option<SessionEvent> {
        while let Some(event) = self.next_event().await {
            if event.kind == kind {
                return Some(event);
            }
        }
        None
    }

    /// Waits until somebody on the call has been heard, which on a phone call is the caller
    /// having picked up and said something.
    pub async fn wait_for_participant(&self) -> Option<SessionEvent> {
        while let Some(event) = self.next_event().await {
            if matches!(event.kind.as_str(), "participant_joined" | "heard") {
                return Some(event);
            }
        }
        None
    }

    /// Speaks text without going through the model, for when you know what should be said.
    pub async fn say(&self, text: &str) -> Result<()> {
        self.command(json!({"type": "say", "text": text})).await
    }

    /// Answers text through the model, as though it had been said on the call.
    ///
    /// [`Responses::create`] is the same thing with an id back.
    pub async fn respond(&self, text: &str) -> Result<()> {
        self.command(json!({"type": "respond", "text": text})).await
    }

    /// Abandons the reply being spoken.
    pub async fn interrupt(&self) -> Result<()> {
        self.command(json!({"type": "interrupt"})).await
    }

    /// Changes what the agent is told to be, from the next turn.
    pub async fn set_instructions(&self, instructions: &str) -> Result<()> {
        self.command(json!({"type": "instructions", "instructions": instructions}))
            .await
    }

    /// Continues this conversation as a new one.
    ///
    /// The parent is untouched and keeps its transcript. `response_id` branches from the end
    /// of that response rather than from where the parent is now, which is also how a
    /// conversation kept in Stream Chat is taken back to an earlier point. The fork runs this
    /// session's functions: a conversation continued without them would offer the model
    /// tools nothing can run.
    pub async fn fork(&self, request: &types::ForkSessionRequest) -> Result<Session> {
        let forked = self.client.fork_session(self.id(), Some(request)).await?;
        Session::watching(
            &self.client,
            forked,
            self.tools.clone(),
            WatchOptions::default(),
        )
        .await
    }

    /// Resolves once the conversation has ended.
    pub async fn wait(&self) {
        let mut ended = self.ended.clone();
        let _ = ended.wait_for(|ended| *ended).await;
    }

    /// Ends the conversation. Safe to call after it has already ended.
    pub async fn close(&self) {
        if self.sender.open() {
            let _ = self.sender.send(&json!({"type": "close"})).await;
        } else if self.live() {
            let _ = self.client.close_session(self.id()).await;
        }
        if tokio::time::timeout(CLOSE_GRACE, self.wait())
            .await
            .is_err()
        {
            self.cancel.cancel();
        }
        let watcher = self.watcher.lock().expect("watcher").take();
        if let Some(watcher) = watcher {
            let _ = watcher.await;
        }
    }

    /// Runs `scope` with the session, then closes it whether `scope` succeeded or not: the
    /// `async with` of the other SDKs.
    ///
    /// ```no_run
    /// # async fn example(agent: vision_agents::Agent) -> vision_agents::Result<()> {
    /// agent.join("my-call").await?.within(async |session| {
    ///     session.responses.create("greet the user in one short sentence").await?;
    ///     Ok(())
    /// }).await
    /// # }
    /// ```
    pub async fn within<T>(self, scope: impl AsyncFnOnce(&Session) -> Result<T>) -> Result<T> {
        let answered = scope(&self).await;
        self.close().await;
        answered
    }

    async fn command(&self, frame: Value) -> Result<()> {
        if !self.sender.open() {
            return Err(Error::Closed(format!("the session {}", self.id())));
        }
        self.sender.send(&frame).await
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Session")
            .field("id", &self.created.id)
            .field("live", &self.live())
            .finish()
    }
}

/// Reads the socket until the conversation ends, answering tool calls as they arrive and
/// queueing everything else for whoever reads events.
///
/// It runs whether or not anybody is reading, because a tool call the model is waiting on
/// cannot depend on the caller having started a loop.
async fn watch_socket(
    mut receiver: SocketReceiver,
    sender: SocketSender,
    shared: Arc<Shared>,
    tools: Tools,
    cancel: CancellationToken,
    ended: watch::Sender<bool>,
) {
    loop {
        let incoming = tokio::select! {
            _ = cancel.cancelled() => {
                // Dropped or given up on: end the conversation rather than leave the backend
                // holding a call nobody is listening to.
                if sender.open() {
                    let _ = sender.send(&json!({"type": "close"})).await;
                }
                break;
            }
            incoming = receiver.next() => incoming,
        };
        let frame = match incoming {
            Some(Ok(Incoming::Frame(frame))) => frame,
            Some(Ok(Incoming::Audio(_))) => continue,
            Some(Err(_)) | None => break,
        };

        match frame.kind() {
            "tool_call" => run_tool(frame, &sender, &shared, &tools),
            "tool_cancel" => {
                if let Some(running) = shared
                    .running
                    .lock()
                    .expect("tools")
                    .remove(frame.text("id"))
                {
                    running.abort();
                }
            }
            _ => {
                let mut events = shared.events.lock().expect("events");
                events.push_back(SessionEvent::of(frame));
                if events.len() > BUFFERED_EVENTS {
                    events.pop_front();
                }
                drop(events);
                shared.arrived.notify_one();
            }
        }
    }

    sender.close().await;
    for (_, running) in shared.running.lock().expect("tools").drain() {
        running.abort();
    }
    let _ = ended.send(true);
    shared.arrived.notify_one();
}

/// Runs one of the caller's functions on its own task and answers the model with what it
/// said. Not awaited, because reading the socket is also what delivers `tool_cancel`.
fn run_tool(frame: Frame, sender: &SocketSender, shared: &Arc<Shared>, tools: &Tools) {
    let id = frame.text("id").to_string();
    let mut result = json!({"type": "tool_result", "tool_call_id": id});
    // A durable command's result is only accepted back with the command and turn it names.
    for key in ["command_id", "turn_id"] {
        if !frame.text(key).is_empty() {
            result[key] = json!(frame.text(key));
        }
    }

    let mut running = shared.running.lock().expect("tools");
    if running.len() >= RUNNING_TOOLS {
        drop(running);
        result["error"] = json!("too many tools are already running");
        let sender = sender.clone();
        tokio::spawn(async move { sender.send(&result).await });
        return;
    }

    let (sender, shared_for_task, tools) = (sender.clone(), shared.clone(), tools.clone());
    let key = id.clone();
    let task = tokio::spawn(async move {
        match tools
            .call(frame.text("name"), frame.text("arguments"))
            .await
        {
            Ok(output) => result["output"] = output,
            Err(error) => result["error"] = json!(error),
        }
        shared_for_task.running.lock().expect("tools").remove(&key);
        if sender.open() {
            let _ = sender.send(&result).await;
        }
    });
    running.insert(id, task.abort_handle());
}
