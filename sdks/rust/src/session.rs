use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
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
use crate::stream::{Call, DEFAULT_CALL_TYPE};
use crate::tool_call::{Holds, Line, ToolCallContext, to_run};
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
    holds: Holds,
    session_id: String,
    changes: Mutex<Changes>,
}

impl Shared {
    /// The functions that run the tool called `name`: the ones [`Session::add_tools`] added
    /// it in, none once [`Session::remove_tool`] withdrew it, else the session's own if they
    /// hold it.
    fn functions_for(&self, name: &str, own: &Tools) -> Option<Tools> {
        let changes = self.changes.lock().expect("tools");
        if let Some(added) = changes.added.get(name) {
            return Some(added.clone());
        }
        (!changes.withdrawn.contains(name) && own.has(name)).then(|| own.clone())
    }
}

/// What [`Session::add_tools`] and [`Session::remove_tool`] changed of the session's tools.
#[derive(Default)]
struct Changes {
    /// The functions added after the session opened, by name, each in the registry it was
    /// added in; they run before the session's own of the same name.
    added: HashMap<String, Tools>,
    /// The names withdrawn from the session, which its own no longer run.
    withdrawn: HashSet<String>,
    /// The names whose last change the router has neither made nor refused, which it may or
    /// may not offer: one in flight, or one whose answer was lost, never waited for, or said
    /// neither (a 5xx, which a gateway can answer after the change was made).
    unsettled: HashSet<String>,
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
    voice: AtomicBool,
    /// Runs one change of the session's tools at a time, across its request.
    changing: tokio::sync::Mutex<()>,
}

impl Session {
    /// Creates a session and starts watching it.
    ///
    /// It returns once the backend is holding it, so a session that has opened is already
    /// listening. Without `start_voice` it is held in writing. The tools are declared on the
    /// request so the model is offered them.
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
                // so it is stopped rather than left holding a call nobody is listening to.
                let _ = client.stop_session(&created.id).await;
                return Err(error);
            }
        };

        let shared = Arc::new(Shared {
            session_id: created.id.clone(),
            ..Default::default()
        });
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
            voice: AtomicBool::new(!created.call_id.is_empty()),
            created,
            client: client.clone(),
            tools,
            sender: socket.sender,
            shared,
            ended,
            cancel,
            watcher: Mutex::new(Some(watcher)),
            changing: tokio::sync::Mutex::new(()),
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

    /// The Stream call the conversation is on while voice is started, named after the
    /// session.
    pub fn call(&self) -> Call {
        Call {
            id: self.id().into(),
            kind: DEFAULT_CALL_TYPE.into(),
        }
    }

    /// Has the agent join the session's call and carry the conversation on there. Starting
    /// voice that is already on does nothing.
    pub async fn start_voice(&self) -> Result<types::Session> {
        let started = self.client.start_session_voice(self.id(), None).await?;
        self.voice.store(true, Ordering::Relaxed);
        Ok(started)
    }

    /// Takes the agent off the call and carries the conversation on in writing. Stopping
    /// voice that is off does nothing.
    pub async fn stop_voice(&self) -> Result<types::Session> {
        let stopped = self.client.stop_session_voice(self.id()).await?;
        self.voice.store(false, Ordering::Relaxed);
        Ok(stopped)
    }

    /// Whether the agent is on the call, as this process last saw it.
    pub fn voice_started(&self) -> bool {
        self.voice.load(Ordering::Relaxed)
    }

    /// Whether the conversation is still being held.
    pub fn live(&self) -> bool {
        !*self.ended.borrow()
    }

    /// The functions this conversation runs for the model.
    pub fn tools(&self) -> &Tools {
        &self.tools
    }

    /// Gives this running session the functions in `tools`, for this session alone, and runs
    /// their calls here. One named as a tool the session already has is refused: to change
    /// one, [`Session::remove_tool`] it first, so that no call runs half under each. The
    /// functions are kept as they are now: registering in `tools` afterwards changes nothing
    /// here.
    ///
    /// The router waits for the turn in progress and every call in flight to finish, and the
    /// model is offered them from the next turn, so do not call this from inside one of this
    /// session's functions. Only the kind of caller that opened the session may, and one
    /// change of its tools runs at a time.
    pub async fn add_tools(&self, tools: Tools) -> Result<()> {
        // A copy, so that registering in the caller's registry afterwards cannot change what
        // a name this session runs does.
        let copy = Tools::new();
        tools.copy_into(&copy, |_| true);
        let declared = copy.declared();
        if declared.is_empty() {
            return Err(Error::configuration("there are no tools to add"));
        }
        let _changing = self.changing.lock().await;

        // A name whose adding was never answered may or may not be the router's: it is
        // withdrawn first, so the router and this process agree on which function it is.
        for tool in &declared {
            if self.unsettled(&tool.name) {
                self.withdraw(&tool.name).await?;
            }
        }

        // Run from before the router answers: the next turn can call one before the answer is
        // here. A name already run here keeps its function, since the router refuses to add
        // it again. Each is unsettled until the answer is here, since a caller who stops
        // waiting for it runs nothing after the request.
        let mut fresh = Vec::new();
        {
            let mut changes = self.shared.changes.lock().expect("tools");
            for tool in &declared {
                if !changes.added.contains_key(&tool.name)
                    && (changes.withdrawn.contains(&tool.name) || !self.tools.has(&tool.name))
                {
                    changes.added.insert(tool.name.clone(), copy.clone());
                    changes.unsettled.insert(tool.name.clone());
                    fresh.push(tool.name.clone());
                }
            }
        }

        let request = types::AddSessionToolsRequest {
            tools: Some(declared),
        };
        let added = self.client.add_session_tools(self.id(), &request).await;
        // Without a success or a refusal the router may have added them, so their calls are
        // still run here until a later change of the name settles it.
        let refused = added.as_ref().is_err_and(is_refusal);
        if added.is_ok() || refused {
            let mut changes = self.shared.changes.lock().expect("tools");
            for name in fresh {
                changes.unsettled.remove(&name);
                if refused {
                    changes.added.remove(&name);
                }
            }
        }
        added
    }

    /// Withdraws one of this session's tools, whether declared when it opened or added
    /// since. The router waits as it does for [`Session::add_tools`], so once this returns no
    /// call of the tool is in flight and none can be made; do not call it from inside one of
    /// this session's functions either.
    pub async fn remove_tool(&self, name: &str) -> Result<()> {
        let _changing = self.changing.lock().await;
        self.withdraw(name).await
    }

    /// Takes the tool called `name` from the session and from what runs here. For a name
    /// whose last change was never answered, the router having no tool by it is as good as
    /// withdrawing it.
    async fn withdraw(&self, name: &str) -> Result<()> {
        // Unsettled until the router answers: it may withdraw the tool though the answer never
        // gets here, and a caller who stops waiting runs nothing after the request.
        let was_unsettled = !self
            .shared
            .changes
            .lock()
            .expect("tools")
            .unsettled
            .insert(name.into());
        let withdrawn = self.client.remove_session_tool(self.id(), name).await;
        let mut changes = self.shared.changes.lock().expect("tools");
        match withdrawn {
            Ok(()) => {}
            Err(error) if was_unsettled && error.status() == Some(404) => {}
            // Refused: the router still has the tool as it was.
            Err(error) if is_refusal(&error) => {
                if !was_unsettled {
                    changes.unsettled.remove(name);
                }
                return Err(error);
            }
            // No answer, or one that does not say whether it was withdrawn: its function still
            // runs here until a later change of the name settles it.
            Err(error) => return Err(error),
        }
        changes.added.remove(name);
        changes.unsettled.remove(name);
        changes.withdrawn.insert(name.into());
        Ok(())
    }

    fn unsettled(&self, name: &str) -> bool {
        self.shared
            .changes
            .lock()
            .expect("tools")
            .unsettled
            .contains(name)
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

    /// Abandons the reply being spoken.
    pub async fn interrupt(&self) -> Result<()> {
        self.command(json!({"type": "interrupt"})).await
    }

    /// Changes this session: its title, description, custom labels, pin, models or voice,
    /// from the next turn. A field left `None` is left as it is. Returns the session
    /// as it now is.
    pub async fn update(&self, update: &types::UpdateSessionRequest) -> Result<types::Session> {
        self.client.update_session(self.id(), update).await
    }

    /// Pins this conversation and returns it as it now is, with `pinned_at` set. A pin does
    /// not move it in the order of last answers.
    pub async fn pin(&self) -> Result<types::Session> {
        self.pinned(true).await
    }

    /// Unpins this conversation and returns it as it now is.
    pub async fn unpin(&self) -> Result<types::Session> {
        self.pinned(false).await
    }

    async fn pinned(&self, pinned: bool) -> Result<types::Session> {
        let update = types::UpdateSessionRequest {
            pinned: Some(pinned),
            ..Default::default()
        };
        self.client.update_session(self.id(), &update).await
    }

    /// Deletes this conversation: it is stopped, and its turns and what it remembered are
    /// deleted with it.
    pub async fn delete(&self) -> Result<()> {
        self.client.delete_session(self.id()).await
    }

    /// Deletes what this conversation remembered, leaving the user's other memories alone.
    /// Server side only.
    pub async fn delete_memories(&self) -> Result<()> {
        self.client.delete_session_memories(self.id()).await
    }

    /// Continues this conversation as a new one.
    ///
    /// The parent is untouched and keeps its transcript. `response_id` branches from the end
    /// of that response rather than from where the parent is now, which is also how a
    /// conversation kept in Stream Chat is taken back to an earlier point. The fork runs this
    /// session's functions as they are now, as its own: the ones it opened with less those
    /// [`Session::remove_tool`] withdrew, and those [`Session::add_tools`] added, since the
    /// router gives a running session's fork its tools as they are. A conversation continued
    /// without them would offer the model tools nothing can run. A change of the tools in
    /// progress is waited for, and the router finishes one only after the turn in progress
    /// and every call in flight, so do not call this from inside one of this session's
    /// functions while one may be in progress.
    pub async fn fork(&self, request: &types::ForkSessionRequest) -> Result<Session> {
        // Copied and asked for under the change lock, so the fork runs the tools the router
        // copies.
        let changing = self.changing.lock().await;
        let tools = self.current_tools();
        // A name whose last change the router neither made nor refused may or may not be
        // among them there, so the fork withdraws it before changing it, as this session would.
        let unsettled = self.shared.changes.lock().expect("tools").unsettled.clone();
        let forked = self.client.fork_session(self.id(), Some(request)).await?;
        drop(changing);
        let fork = Session::watching(&self.client, forked, tools, WatchOptions::default()).await?;
        fork.shared.changes.lock().expect("tools").unsettled = unsettled;
        Ok(fork)
    }

    /// A registry of the functions that run this session's tools now.
    fn current_tools(&self) -> Tools {
        let current = Tools::new();
        let changes = self.shared.changes.lock().expect("tools");
        self.tools
            .copy_into(&current, |name| !changes.withdrawn.contains(name));
        for (name, added) in &changes.added {
            added.copy_into(&current, |each| each == name);
        }
        current
    }

    /// Resolves once the conversation has ended.
    pub async fn wait(&self) {
        let mut ended = self.ended.clone();
        let _ = ended.wait_for(|ended| *ended).await;
    }

    /// Stops the conversation. Safe to call after it has already ended. What it recorded and
    /// remembered is kept; [`Session::delete`] takes it away.
    pub async fn close(&self) {
        if self.sender.open() {
            let _ = self.sender.send(&json!({"type": "close"})).await;
        } else if self.live() {
            let _ = self.client.stop_session(self.id()).await;
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
    /// agent.join().await?.within(async |session| {
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
            // Every watcher is sent every call and the first answer wins: answering a tool
            // held elsewhere would beat whoever runs it, with an error.
            "tool_call" => match shared.functions_for(frame.text("name"), &tools) {
                Some(functions) if to_run(&frame) => run_tool(frame, &sender, &shared, &functions),
                _ => {}
            },
            // A call held through interruptions is not cancelled by one: only its timeout
            // and the end of the session stop it.
            "tool_cancel" => {
                let id = frame.text("id");
                shared.holds.answer(id, false);
                if !shared.holds.held(id)
                    && let Some(running) = shared.running.lock().expect("tools").remove(id)
                {
                    running.abort();
                }
            }
            kind => {
                if kind == "tool_held" {
                    shared.holds.answer(frame.text("id"), true);
                }
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
    // What add_tools and remove_tool changed was this conversation's alone.
    *shared.changes.lock().expect("tools") = Changes::default();
    let _ = ended.send(true);
    shared.arrived.notify_one();
}

/// Whether the router said no to a change of a session's tools, leaving them as they were.
/// Only a 4xx says so: a gateway in front of it can answer 502 or 504 after the change was
/// made, so any other status, like no answer, leaves the change unknown.
fn is_refusal(error: &Error) -> bool {
    matches!(error.status(), Some(400..=499))
}

/// Runs one of the caller's functions on its own task and answers the model with what it
/// said. Not awaited, because reading the socket is also what delivers `tool_cancel`.
fn run_tool(frame: Frame, sender: &SocketSender, shared: &Arc<Shared>, tools: &Tools) {
    let id = frame.text("id").to_string();
    let mut result = json!({"type": "tool_result", "tool_call_id": id});
    // A durable command's result is only accepted back with the request and turn it names.
    for key in ["request_id", "turn_id"] {
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

    let line = Line {
        sender: sender.clone(),
        holds: shared.holds.clone(),
        key: id.clone(),
    };
    let tool_call_context = ToolCallContext::of(&frame, &shared.session_id, line);
    let (sender, shared_for_task, tools) = (sender.clone(), shared.clone(), tools.clone());
    let key = id.clone();
    let task = tokio::spawn(async move {
        match tools
            .call_with_tool_call_context(
                frame.text("name"),
                frame.text("arguments"),
                tool_call_context,
            )
            .await
        {
            Ok(output) => result["output"] = output,
            Err(error) => result["error"] = json!(error),
        }
        shared_for_task.running.lock().expect("tools").remove(&key);
        shared_for_task.holds.forget(&key);
        if sender.open() {
            let _ = sender.send(&result).await;
        }
    });
    running.insert(id, task.abort_handle());
}
