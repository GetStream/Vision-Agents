use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};
use tokio::sync::oneshot;

use crate::error::{Error, Result};
use crate::socket::{Frame, SocketSender};

/// Who a tool call answers, as the router reports them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolUser {
    pub id: String,
    pub name: String,
    pub custom: Map<String, Value>,
}

/// The conversation a tool call happens in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolConversation {
    pub cid: String,
    pub custom: Map<String, Value>,
}

/// One call of one of the caller's functions: who it is for, and its line back to them.
///
/// A function registered with
/// [`Tools::register_with_tool_call_context`](crate::Tools::register_with_tool_call_context)
/// is handed one. A call made directly, as a test does, has no line, so its updates go nowhere.
#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    pub id: String,
    pub session_id: String,
    pub turn_id: String,
    pub user: ToolUser,
    pub conversation: ToolConversation,
    /// Whether a person allowed this call before it ran.
    pub approved: bool,
    line: Option<Line>,
}

impl ToolCall {
    /// The call a `tool_call` frame asks for. `session_id` stands in for a frame that does
    /// not name its session.
    pub(crate) fn of(frame: &Frame, session_id: &str, line: Line) -> Self {
        let user = frame.nested("user").unwrap_or_default();
        let conversation = frame.nested("conversation").unwrap_or_default();
        ToolCall {
            id: frame.text("id").into(),
            session_id: match frame.text("session_id") {
                "" => session_id.into(),
                named => named.into(),
            },
            turn_id: frame.text("turn_id").into(),
            user: ToolUser {
                id: user.text("id").into(),
                name: user.text("name").into(),
                custom: user.nested("custom").unwrap_or_default().0,
            },
            conversation: ToolConversation {
                cid: conversation.text("cid").into(),
                custom: conversation.nested("custom").unwrap_or_default().0,
            },
            approved: frame.flag("approved"),
            line: Some(line),
        }
    }

    /// Tells the people in the conversation how the call is going, such as "Found three
    /// flights, comparing prices". The agent puts it in its own words.
    pub async fn update(&self, text: &str) -> Result<()> {
        self.send_update(text, false).await
    }

    /// [`ToolCall::update`], said word for word.
    pub async fn say(&self, text: &str) -> Result<()> {
        self.send_update(text, true).await
    }

    /// Keeps the call going if the person talks over the agent from here on, for a call
    /// that is past the point of taking back, such as one that has started a payment.
    ///
    /// The router has the last word: a call it already cancelled is not held, and this
    /// returns [`Error::NotHeld`]. A held call still ends at its timeout, or with the session.
    pub async fn disallow_interruptions(&self) -> Result<()> {
        let Some(line) = &self.line else {
            return Ok(());
        };
        if line.holds.held(&line.key) {
            return Ok(());
        }
        let answer = line.holds.ask(&line.key);
        line.sender
            .send(&json!({"type": "tool_hold", "tool_call_id": self.id}))
            .await?;
        match answer.await {
            Ok(true) => Ok(()),
            _ => Err(Error::NotHeld(self.id.clone())),
        }
    }

    async fn send_update(&self, text: &str, verbatim: bool) -> Result<()> {
        let Some(line) = &self.line else {
            return Ok(());
        };
        line.sender
            .send(&json!({"type": "tool_update", "tool_call_id": self.id, "text": text, "verbatim": verbatim}))
            .await
    }
}

/// A call's way back over the socket that delivered it.
#[derive(Clone)]
pub(crate) struct Line {
    pub(crate) sender: SocketSender,
    pub(crate) holds: Holds,
    /// What the socket's `tool_cancel` names this call by.
    pub(crate) key: String,
}

impl std::fmt::Debug for Line {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Line")
    }
}

/// The calls that asked to be held, waiting on the router's `tool_held` or `tool_cancel`,
/// and the ones it held. Keyed by whatever the socket's `tool_cancel` names a call by.
#[derive(Clone, Default)]
pub(crate) struct Holds {
    inner: Arc<Mutex<HoldsInner>>,
}

#[derive(Default)]
struct HoldsInner {
    asking: HashMap<String, oneshot::Sender<bool>>,
    held: HashSet<String>,
}

impl Holds {
    fn ask(&self, key: &str) -> oneshot::Receiver<bool> {
        let (answer, answered) = oneshot::channel();
        self.inner
            .lock()
            .expect("holds")
            .asking
            .insert(key.into(), answer);
        answered
    }

    /// Hands the router's answer to the call waiting on it, if one is.
    pub(crate) fn answer(&self, key: &str, held: bool) {
        let mut inner = self.inner.lock().expect("holds");
        if let Some(asking) = inner.asking.remove(key) {
            if held {
                inner.held.insert(key.into());
            }
            let _ = asking.send(held);
        }
    }

    /// Whether the router keeps this call going when the person talks over the agent.
    pub(crate) fn held(&self, key: &str) -> bool {
        self.inner.lock().expect("holds").held.contains(key)
    }

    pub(crate) fn forget(&self, key: &str) {
        let mut inner = self.inner.lock().expect("holds");
        inner.asking.remove(key);
        inner.held.remove(key);
    }
}

/// Whether a `tool_call` frame is one for this process to run: `run` false says the router
/// or another watcher has it.
pub(crate) fn to_run(frame: &Frame) -> bool {
    frame.0.get("run") != Some(&Value::Bool(false))
}
