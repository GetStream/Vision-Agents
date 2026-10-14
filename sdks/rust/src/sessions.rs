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

/// Which of an agent's conversations to list. A field left `None` does not narrow it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Query {
    /// One project's. A search covers every project, so it refuses this.
    pub project_id: Option<String>,
    /// One user's, which only a server-side caller may ask for.
    pub user_id: Option<String>,
    /// How the user took part: `text`, `voice` or `video`.
    pub modality: Option<String>,
    /// The ones still running, `live`, or the ones over, `ended`.
    pub state: Option<String>,
    /// The ones created with this agent id.
    pub agent_id: Option<String>,
    /// Up to 200. `None` is 25.
    pub limit: Option<i64>,
    /// The `next_cursor` of the page before, with the same filters. `None` is the first page.
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

    /// A page of the agent's conversations, most recently updated first, the ones that
    /// ended included. Pass the page's `next_cursor` as [`Query::cursor`] for the next one.
    ///
    /// These are rows rather than live handles: most of them are over.
    pub async fn query(&self, query: Query) -> Result<types::SessionPage> {
        self.client
            .query_sessions(Some(&self.query_of(None, query)))
            .await
    }

    /// Finds a conversation by its title, description and opening question, best match
    /// first. It pages the way [`Sessions::query`] does.
    pub async fn search(&self, text: &str, query: Query) -> Result<types::SessionPage> {
        self.client
            .query_sessions(Some(&self.query_of(Some(text), query)))
            .await
    }

    /// One conversation, whether or not it is still being held.
    pub async fn get(&self, id: &str) -> Result<types::Session> {
        self.client.get_session(id).await
    }

    /// Changes one conversation, whether or not it is still being held, and returns it as it
    /// now is. One that ended can still be renamed and relabelled; models and voice need it
    /// running. A field left `None` is left as it is.
    pub async fn update(
        &self,
        id: &str,
        update: &types::UpdateSessionRequest,
    ) -> Result<types::Session> {
        self.client.update_session(id, update).await
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

    /// The query the listing and the search share, narrowed to this agent. Text makes it a
    /// search.
    fn query_of(&self, text: Option<&str>, query: Query) -> types::SessionQuery {
        let equals = |value: Option<String>| value.map(types::Equals::String);
        types::SessionQuery {
            filter: Some(types::SessionFilter {
                agent: equals(Some(self.agent.clone())),
                project_id: equals(query.project_id),
                user_id: equals(query.user_id),
                modality: equals(query.modality),
                state: equals(query.state),
                agent_id: equals(query.agent_id),
                text: text.map(|text| types::TextMatch { q: text.into() }),
                ..Default::default()
            }),
            limit: query.limit,
            cursor: query.cursor,
            sort: None,
        }
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
