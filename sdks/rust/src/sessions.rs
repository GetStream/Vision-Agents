use reqwest::Method;
use serde_json::{Map, Value, json};

use crate::client::Client;
use crate::error::{Error, Result};
use crate::operations::ListAgentConfigsQuery;
use crate::responses::Responses;
use crate::session::{Session, WatchOptions};
use crate::tools::Tools;
use crate::types;

/// An agent, addressed by the name it is configured under.
///
/// A handle rather than a description: what the agent is was decided once, in the backend,
/// which is the point of naming it rather than spelling it out per conversation. A name that
/// matches nothing is refused when a session is opened rather than here, because this costs
/// no request.
#[derive(Debug, Clone)]
pub struct AgentRef {
    /// What the agent is called.
    pub name: String,
    /// This agent's conversations: opening one, and reading the old ones back.
    pub sessions: Sessions,
    /// Functions run for every session of this agent, whoever opened it, once a dispatch
    /// worker hosts them with [`Dispatch::host`](crate::Dispatch::host).
    pub tools: Tools,
    client: Client,
}

impl AgentRef {
    pub(crate) fn new(client: Client, name: &str) -> Self {
        AgentRef {
            name: name.into(),
            sessions: Sessions {
                client: client.clone(),
                agent: name.into(),
            },
            tools: Tools::new(),
            client,
        }
    }

    /// How the agent is configured, as the backend has it. Server side only.
    pub async fn config(&self) -> Result<Option<types::AgentConfig>> {
        let query = ListAgentConfigsQuery {
            name: Some(self.name.clone()),
            role: None,
        };
        let stored = self.client.list_agent_configs(&query).await?;
        Ok(stored.into_iter().find(|config| config.name == self.name))
    }

    /// Changes some of how the agent is configured and returns the config as it now is. A
    /// field left out of the patch keeps what is stored. Server side only.
    pub async fn update_config(
        &self,
        patch: &types::AgentConfigPatch,
    ) -> Result<types::AgentConfig> {
        let Some(config) = self.config().await? else {
            return Err(Error::configuration(format!(
                "there is no agent called {} to update",
                self.name
            )));
        };
        self.client.patch_agent_config(&config.id, patch).await
    }
}

/// Which of an agent's conversations to list, in the shape the router takes it.
///
/// ```no_run
/// # use serde_json::json;
/// # use vision_agents::{Query, types};
/// let query = Query {
///     filter: json!({"$or": [{"project_id": "web"}, {"project_id": "ios"}]}),
///     sort: vec![types::SessionSort {
///         field: types::SessionSortField::Pinned,
///         direction: None,
///     }],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Query {
    /// Which sessions, as a JSON object with the router's own key names. The keys of one
    /// object are ANDed, and `$and` and `$or` each take 1 to 10 filters, up to 3 deep.
    ///
    /// - `project_id`: `"p"`, `{"$eq": "p"}` or `{"$in": ["p", "q"]}`.
    /// - `agent`: always this agent's name at the top level, whatever is put there.
    /// - `text`: `{"$q": "words"}`, once, at the top level or in the top-level `$and`, never
    ///   inside `$or`. [`Sessions::search`] sets it.
    /// - `user_id`: `"u"` or `{"$eq": "u"}`, at the top level or in the top-level `$and`.
    ///   Server side only.
    /// - `review_score`: a number from 1 to 5, or an object of `$eq`, `$gt`, `$gte`, `$lt`
    ///   and `$lte`. Server side only.
    ///
    /// The router refuses any other key. `Value::Null`, the default, does not narrow it.
    pub filter: Value,
    /// The order. Empty is the most recently answered first, or the best match first for a
    /// search. Otherwise one of: `LastMessageAt`; `Pinned`, optionally followed by
    /// `LastMessageAt`, which lists the pinned sessions first; `ReviewScore`, server side
    /// only; `Relevance`, for a search. A direction left `None` is descending.
    pub sort: Vec<types::SessionSort>,
    /// Up to 200. `None` is 25.
    pub limit: Option<u32>,
    /// The `next_cursor` of the page before, with the same filter and sort. `None` is the
    /// first page.
    pub cursor: Option<String>,
}

/// An agent's conversations: the one being held and the ones that were.
#[derive(Debug, Clone)]
pub struct Sessions {
    client: Client,
    agent: String,
}

impl Sessions {
    /// Opens a conversation and starts watching it.
    ///
    /// The agent is always this one. Without `start_voice` it is held in writing, and
    /// [`Session::start_voice`] puts the agent on the session's call later.
    pub async fn create(
        &self,
        mut request: types::CreateSessionRequest,
        tools: Tools,
    ) -> Result<Session> {
        request.agent = Some(self.agent.clone());
        request.config_id = None;
        Session::open(&self.client, request, tools, WatchOptions::default()).await
    }

    /// Carries on a conversation held in writing, by the id of the session it was held in,
    /// and starts watching it. One that ended is reopened with what was said in it.
    pub async fn resume(&self, id: &str, tools: Tools) -> Result<Session> {
        let got = self.get(id).await?;
        Session::watching(&self.client, got, tools, WatchOptions::default()).await
    }

    /// A page of the agent's conversations, the ones that ended included, most recently
    /// answered first unless [`Query::sort`] says otherwise. Pass the page's `next_cursor` as
    /// [`Query::cursor`] for the next one.
    ///
    /// These are rows rather than live handles: most of them are over.
    pub async fn query(&self, query: Query) -> Result<types::SessionPage> {
        self.send(self.body_of(None, query)?).await
    }

    /// Finds a conversation by its title, description, project and agent name, best match
    /// first. It takes the filter and pages the way [`Sessions::query`] does; the text is the
    /// filter's `text`, so a filter that already has one is refused.
    pub async fn search(&self, text: &str, query: Query) -> Result<types::SessionPage> {
        self.send(self.body_of(Some(text), query)?).await
    }

    /// One conversation, whether or not it is still being held.
    pub async fn get(&self, id: &str) -> Result<types::Session> {
        self.client.get_session(id).await
    }

    /// Changes one conversation, whether or not it is still being held, and returns it as it
    /// now is. One that ended can still be renamed, relabelled, pinned and unpinned; models
    /// and voice need it running. A field left `None` is left as it is.
    pub async fn update(
        &self,
        id: &str,
        update: &types::UpdateSessionRequest,
    ) -> Result<types::Session> {
        self.client.update_session(id, update).await
    }

    /// Pins a conversation, running or ended, and returns it as it now is, with `pinned_at`
    /// set. One already pinned keeps the time it was first pinned.
    pub async fn pin(&self, id: &str) -> Result<types::Session> {
        self.pinned(id, true).await
    }

    /// Unpins a conversation and returns it as it now is, without `pinned_at`.
    pub async fn unpin(&self, id: &str) -> Result<types::Session> {
        self.pinned(id, false).await
    }

    async fn pinned(&self, id: &str, pinned: bool) -> Result<types::Session> {
        let update = types::UpdateSessionRequest {
            pinned: Some(pinned),
            ..Default::default()
        };
        self.client.update_session(id, &update).await
    }

    /// Deletes a conversation, running or ended: it is stopped, and its turns and what it
    /// remembered are deleted with it. The user's other memories are kept.
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.client.delete_session(id).await
    }

    /// Deletes what one conversation remembered, running or ended, and leaves the rest of
    /// the user's memories alone. Server side only.
    pub async fn delete_memories(&self, id: &str) -> Result<()> {
        self.client.delete_session_memories(id).await
    }

    /// The turns of a conversation this process is not holding.
    pub fn responses(&self, id: &str) -> Responses {
        Responses::new(self.client.clone(), id)
    }

    /// The body the listing and the search share. The caller's filter is copied key by key
    /// and narrowed to this agent at its top level, never wrapped, because `text` and
    /// `user_id` only count there. Text makes it a search.
    fn body_of(&self, text: Option<&str>, query: Query) -> Result<Value> {
        let mut filter = match query.filter {
            Value::Null => Map::new(),
            Value::Object(filter) => filter,
            other => {
                return Err(Error::configuration(format!(
                    "a session filter is a JSON object, not {other}"
                )));
            }
        };
        filter.insert("agent".into(), self.agent.clone().into());
        if let Some(text) = text {
            if filter.contains_key("text") {
                return Err(Error::configuration(
                    "search sets the filter's text itself, so the filter cannot have one",
                ));
            }
            filter.insert("text".into(), json!({"$q": text}));
        }

        let mut body = Map::new();
        body.insert("filter".into(), filter.into());
        if !query.sort.is_empty() {
            body.insert("sort".into(), json!(query.sort));
        }
        if let Some(limit) = query.limit {
            body.insert("limit".into(), limit.into());
        }
        if let Some(cursor) = query.cursor {
            body.insert("cursor".into(), cursor.into());
        }
        Ok(body.into())
    }

    /// Sends a query as JSON rather than through the generated filter type, so a key newer
    /// than this build is the router's to accept or refuse.
    async fn send(&self, body: Value) -> Result<types::SessionPage> {
        self.client
            .send(
                Method::POST,
                "/v1/agents/sessions/query",
                None::<&()>,
                Some(&body),
                "querySessions",
            )
            .await
    }
}

/// What agents remember about the app's users between conversations.
#[derive(Debug, Clone)]
pub struct Memories {
    pub(crate) client: Client,
}

impl Memories {
    /// Deletes everything remembered about one user: every session's and every agent's.
    /// `user_id` is the `user_id` of the memory filter the sessions were opened with. Server
    /// side only.
    pub async fn truncate(&self, user_id: &str) -> Result<()> {
        if user_id.is_empty() {
            return Err(Error::configuration("truncating memories needs a user id"));
        }
        self.client.truncate_memories(user_id).await
    }
}
