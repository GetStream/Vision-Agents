use crate::client::Client;
use crate::error::Result;
use crate::operations::{ListAgentConfigsQuery, ListSessionsQuery, SearchSessionsQuery};
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
    /// The agent is always this one. Without a `call_id` it is held in writing: a session
    /// resource is a conversation, and a caller who wants one on a call says which call.
    pub async fn create(
        &self,
        mut request: types::CreateSessionRequest,
        tools: Tools,
    ) -> Result<Session> {
        request.agent = Some(self.agent.clone());
        request.config_id = None;
        if request.call_id.is_none() && request.text.is_none() {
            request.text = Some(true);
        }
        Session::open(&self.client, request, tools, WatchOptions::default()).await
    }

    /// The agent's conversations, newest first, the ones that ended included.
    ///
    /// These are rows rather than live handles: most of them are over.
    pub async fn query(&self, mut query: ListSessionsQuery) -> Result<Vec<types::Session>> {
        query.agent = Some(self.agent.clone());
        self.client.list_sessions(&query).await
    }

    /// Finds a conversation by its title, description and opening question.
    pub async fn search(
        &self,
        text: &str,
        query: ListSessionsQuery,
    ) -> Result<Vec<types::Session>> {
        let query = SearchSessionsQuery {
            q: Some(text.into()),
            agent: Some(self.agent.clone()),
            config_id: query.config_id,
            user_id: query.user_id,
            project: query.project,
            state: query.state,
            custom: query.custom,
            created_after: query.created_after,
            created_before: query.created_before,
            limit: query.limit,
            offset: query.offset,
        };
        self.client.search_sessions(&query).await
    }

    /// One conversation, whether or not it is still being held.
    pub async fn get(&self, id: &str) -> Result<types::Session> {
        self.client.get_session(id).await
    }

    /// The turns of a conversation this process is not holding.
    pub fn responses(&self, id: &str) -> Responses {
        Responses::new(self.client.clone(), id)
    }
}
