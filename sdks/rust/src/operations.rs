//! One method per HTTP operation in the spec.
#![allow(clippy::too_many_arguments)]
use crate::client::{Client, segment};
use crate::error::Result;
use crate::types;
use reqwest::Method;
use serde::Serialize;
impl Client {
    /// Liveness and dependency check
    ///
    /// `GET /health` (`getHealth`).
    pub async fn get_health(&self) -> Result<types::HealthStatus> {
        self.send(
            Method::GET,
            "/health",
            None::<&()>,
            None::<&()>,
            "getHealth",
        )
        .await
    }
    /// The calls the calling customer has run
    ///
    /// `GET /v1/agents/calls` (`listCalls`).
    pub async fn list_calls(&self, query: &ListCallsQuery) -> Result<Vec<types::Call>> {
        self.send(
            Method::GET,
            "/v1/agents/calls",
            Some(query),
            None::<&()>,
            "listCalls",
        )
        .await
    }
    /// One call, with whatever was made of it afterwards
    ///
    /// `GET /v1/agents/calls/{id}` (`getCall`).
    pub async fn get_call(&self, id: &str) -> Result<types::Call> {
        self.send(
            Method::GET,
            &format!("/v1/agents/calls/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getCall",
        )
        .await
    }
    /// What the conversation decided, and why
    ///
    /// `GET /v1/agents/calls/{id}/events` (`getCallEvents`).
    pub async fn get_call_events(
        &self,
        id: &str,
        query: &GetCallEventsQuery,
    ) -> Result<Vec<types::CallEvent>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/calls/{id}/events", id = segment(id)),
            Some(query),
            None::<&()>,
            "getCallEvents",
        )
        .await
    }
    /// The call as it unfolded, said and measured together
    ///
    /// `GET /v1/agents/calls/{id}/timeline` (`getCallTimeline`).
    pub async fn get_call_timeline(&self, id: &str) -> Result<Vec<types::TimelineEntry>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/calls/{id}/timeline", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getCallTimeline",
        )
        .await
    }
    /// What a browser needs to join this call
    ///
    /// `POST /v1/agents/calls/{id}/token` (`createCallToken`).
    pub async fn create_call_token(
        &self,
        id: &str,
        body: Option<&types::CallTokenRequest>,
    ) -> Result<types::CallToken> {
        self.send(
            Method::POST,
            &format!("/v1/agents/calls/{id}/token", id = segment(id)),
            None::<&()>,
            body,
            "createCallToken",
        )
        .await
    }
    /// What was said on a call
    ///
    /// `GET /v1/agents/calls/{id}/transcript` (`getCallTranscript`).
    pub async fn get_call_transcript(&self, id: &str) -> Result<Vec<types::TranscriptMessage>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/calls/{id}/transcript", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getCallTranscript",
        )
        .await
    }
    /// The campaigns the calling customer has
    ///
    /// `GET /v1/agents/campaigns` (`listCampaigns`).
    pub async fn list_campaigns(&self) -> Result<Vec<types::Campaign>> {
        self.send(
            Method::GET,
            "/v1/agents/campaigns",
            None::<&()>,
            None::<&()>,
            "listCampaigns",
        )
        .await
    }
    /// Define a list of people to ring
    ///
    /// `POST /v1/agents/campaigns` (`createCampaign`).
    pub async fn create_campaign(&self, body: &types::CampaignRequest) -> Result<types::Campaign> {
        self.send(
            Method::POST,
            "/v1/agents/campaigns",
            None::<&()>,
            Some(body),
            "createCampaign",
        )
        .await
    }
    /// One campaign
    ///
    /// `GET /v1/agents/campaigns/{id}` (`getCampaign`).
    pub async fn get_campaign(&self, id: &str) -> Result<types::Campaign> {
        self.send(
            Method::GET,
            &format!("/v1/agents/campaigns/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getCampaign",
        )
        .await
    }
    /// Who a campaign is ringing, and how far it has got
    ///
    /// `GET /v1/agents/campaigns/{id}/contacts` (`listCampaignContacts`).
    pub async fn list_campaign_contacts(&self, id: &str) -> Result<Vec<types::Contact>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/campaigns/{id}/contacts", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "listCampaignContacts",
        )
        .await
    }
    /// Add people to ring
    ///
    /// `POST /v1/agents/campaigns/{id}/contacts` (`addCampaignContacts`).
    pub async fn add_campaign_contacts(
        &self,
        id: &str,
        body: &types::ContactsRequest,
    ) -> Result<Vec<types::Contact>> {
        self.send(
            Method::POST,
            &format!("/v1/agents/campaigns/{id}/contacts", id = segment(id)),
            None::<&()>,
            Some(body),
            "addCampaignContacts",
        )
        .await
    }
    /// Stop ringing anybody new
    ///
    /// `POST /v1/agents/campaigns/{id}/pause` (`pauseCampaign`).
    pub async fn pause_campaign(&self, id: &str) -> Result<types::Campaign> {
        self.send(
            Method::POST,
            &format!("/v1/agents/campaigns/{id}/pause", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "pauseCampaign",
        )
        .await
    }
    /// Start ringing
    ///
    /// `POST /v1/agents/campaigns/{id}/start` (`startCampaign`).
    pub async fn start_campaign(&self, id: &str) -> Result<types::Campaign> {
        self.send(
            Method::POST,
            &format!("/v1/agents/campaigns/{id}/start", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "startCampaign",
        )
        .await
    }
    /// What a browser needs to read an agent's conversation
    ///
    /// `POST /v1/agents/chat-token` (`createChatToken`).
    pub async fn create_chat_token(
        &self,
        body: &types::ChatTokenRequest,
    ) -> Result<types::ChatToken> {
        self.send(
            Method::POST,
            "/v1/agents/chat-token",
            None::<&()>,
            Some(body),
            "createChatToken",
        )
        .await
    }
    /// The agent configs the calling customer holds
    ///
    /// `GET /v1/agents/configs` (`listAgentConfigs`).
    pub async fn list_agent_configs(
        &self,
        query: &ListAgentConfigsQuery,
    ) -> Result<Vec<types::AgentConfig>> {
        self.send(
            Method::GET,
            "/v1/agents/configs",
            Some(query),
            None::<&()>,
            "listAgentConfigs",
        )
        .await
    }
    /// Store a named configuration a session can be created from
    ///
    /// `POST /v1/agents/configs` (`createAgentConfig`).
    pub async fn create_agent_config(
        &self,
        body: &types::AgentConfigRequest,
    ) -> Result<types::AgentConfig> {
        self.send(
            Method::POST,
            "/v1/agents/configs",
            None::<&()>,
            Some(body),
            "createAgentConfig",
        )
        .await
    }
    /// One agent config
    ///
    /// `GET /v1/agents/configs/{id}` (`getAgentConfig`).
    pub async fn get_agent_config(&self, id: &str) -> Result<types::AgentConfig> {
        self.send(
            Method::GET,
            &format!("/v1/agents/configs/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getAgentConfig",
        )
        .await
    }
    /// Replace an agent config
    ///
    /// `PUT /v1/agents/configs/{id}` (`updateAgentConfig`).
    pub async fn update_agent_config(
        &self,
        id: &str,
        body: &types::AgentConfigRequest,
    ) -> Result<types::AgentConfig> {
        self.send(
            Method::PUT,
            &format!("/v1/agents/configs/{id}", id = segment(id)),
            None::<&()>,
            Some(body),
            "updateAgentConfig",
        )
        .await
    }
    /// Delete an agent config
    ///
    /// `DELETE /v1/agents/configs/{id}` (`deleteAgentConfig`).
    pub async fn delete_agent_config(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/agents/configs/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "deleteAgentConfig",
        )
        .await
    }
    /// The plugin logins this agent holds
    ///
    /// `GET /v1/agents/configs/{id}/plugins` (`listConfigPlugins`).
    pub async fn list_config_plugins(&self, id: &str) -> Result<Vec<types::PluginConnection>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/configs/{id}/plugins", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "listConfigPlugins",
        )
        .await
    }
    /// Drop a plugin login
    ///
    /// `DELETE /v1/agents/configs/{id}/plugins/{plugin_id}` (`disconnectPlugin`).
    pub async fn disconnect_plugin(&self, id: &str, plugin_id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!(
                "/v1/agents/configs/{id}/plugins/{plugin_id}",
                id = segment(id),
                plugin_id = segment(plugin_id)
            ),
            None::<&()>,
            None::<&()>,
            "disconnectPlugin",
        )
        .await
    }
    /// Start a plugin login
    ///
    /// `POST /v1/agents/configs/{id}/plugins/{plugin_id}/authorize` (`authorizePlugin`).
    pub async fn authorize_plugin(
        &self,
        id: &str,
        plugin_id: &str,
        body: Option<&types::AuthorizePluginRequest>,
    ) -> Result<types::PluginAuthorization> {
        self.send(
            Method::POST,
            &format!(
                "/v1/agents/configs/{id}/plugins/{plugin_id}/authorize",
                id = segment(id),
                plugin_id = segment(plugin_id)
            ),
            None::<&()>,
            body,
            "authorizePlugin",
        )
        .await
    }
    /// What a command in this conversation ended as
    ///
    /// `GET /v1/agents/conversations/{cid}/commands/{command_id}` (`getConversationCommand`).
    pub async fn get_conversation_command(
        &self,
        cid: &str,
        command_id: &str,
        query: &GetConversationCommandQuery,
    ) -> Result<types::CommandReceipt> {
        self.send(
            Method::GET,
            &format!(
                "/v1/agents/conversations/{cid}/commands/{command_id}",
                cid = segment(cid),
                command_id = segment(command_id)
            ),
            Some(query),
            None::<&()>,
            "getConversationCommand",
        )
        .await
    }
    /// Read a persistent text conversation
    ///
    /// `GET /v1/agents/conversations/{cid}/messages` (`getConversationMessages`).
    pub async fn get_conversation_messages(
        &self,
        cid: &str,
        query: &GetConversationMessagesQuery,
    ) -> Result<serde_json::Value> {
        self.send(
            Method::GET,
            &format!(
                "/v1/agents/conversations/{cid}/messages",
                cid = segment(cid)
            ),
            Some(query),
            None::<&()>,
            "getConversationMessages",
        )
        .await
    }
    /// Mint a guest so somebody can talk to an agent before signing up
    ///
    /// `POST /v1/agents/guests` (`createGuestUser`).
    pub async fn create_guest_user(
        &self,
        body: Option<&types::GuestUserRequest>,
    ) -> Result<types::GuestUser> {
        self.send(
            Method::POST,
            "/v1/agents/guests",
            None::<&()>,
            body,
            "createGuestUser",
        )
        .await
    }
    /// Move a guest's conversations onto the account they turned out to be
    ///
    /// `POST /v1/agents/guests/claim` (`claimGuestUser`).
    pub async fn claim_guest_user(
        &self,
        body: &types::ClaimGuestRequest,
    ) -> Result<types::ClaimGuestResult> {
        self.send(
            Method::POST,
            "/v1/agents/guests/claim",
            None::<&()>,
            Some(body),
            "claimGuestUser",
        )
        .await
    }
    /// Fill a knowledge base with what the business wrote down
    ///
    /// `POST /v1/agents/knowledge` (`ingestKnowledge`).
    pub async fn ingest_knowledge(
        &self,
        body: &types::IngestKnowledgeRequest,
    ) -> Result<types::IngestedKnowledge> {
        self.send(
            Method::POST,
            "/v1/agents/knowledge",
            None::<&()>,
            Some(body),
            "ingestKnowledge",
        )
        .await
    }
    /// The documents a knowledge base was filled with
    ///
    /// `GET /v1/agents/knowledge/documents` (`listKnowledgeDocuments`).
    pub async fn list_knowledge_documents(
        &self,
        query: &ListKnowledgeDocumentsQuery,
    ) -> Result<Vec<types::IndexedKnowledgeDocument>> {
        self.send(
            Method::GET,
            "/v1/agents/knowledge/documents",
            Some(query),
            None::<&()>,
            "listKnowledgeDocuments",
        )
        .await
    }
    /// One document, with the text it was posted as
    ///
    /// `GET /v1/agents/knowledge/documents/{id}` (`getKnowledgeDocument`).
    pub async fn get_knowledge_document(
        &self,
        id: &str,
    ) -> Result<types::IndexedKnowledgeDocument> {
        self.send(
            Method::GET,
            &format!("/v1/agents/knowledge/documents/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getKnowledgeDocument",
        )
        .await
    }
    /// Take a document out of a knowledge base
    ///
    /// `DELETE /v1/agents/knowledge/documents/{id}` (`deleteKnowledgeDocument`).
    pub async fn delete_knowledge_document(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/agents/knowledge/documents/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "deleteKnowledgeDocument",
        )
        .await
    }
    /// What a document was cut into, in order
    ///
    /// `GET /v1/agents/knowledge/documents/{id}/passages` (`listKnowledgeDocumentPassages`).
    pub async fn list_knowledge_document_passages(
        &self,
        id: &str,
    ) -> Result<Vec<types::KnowledgePassage>> {
        self.send(
            Method::GET,
            &format!(
                "/v1/agents/knowledge/documents/{id}/passages",
                id = segment(id)
            ),
            None::<&()>,
            None::<&()>,
            "listKnowledgeDocumentPassages",
        )
        .await
    }
    /// The pages a knowledge base is kept filled from
    ///
    /// `GET /v1/agents/knowledge/urls` (`listKnowledgeUrls`).
    pub async fn list_knowledge_urls(
        &self,
        query: &ListKnowledgeUrlsQuery,
    ) -> Result<Vec<types::KnowledgeUrl>> {
        self.send(
            Method::GET,
            "/v1/agents/knowledge/urls",
            Some(query),
            None::<&()>,
            "listKnowledgeUrls",
        )
        .await
    }
    /// Keep a knowledge base filled from a page
    ///
    /// `POST /v1/agents/knowledge/urls` (`addKnowledgeUrl`).
    pub async fn add_knowledge_url(
        &self,
        body: &types::KnowledgeUrlRequest,
    ) -> Result<types::KnowledgeUrl> {
        self.send(
            Method::POST,
            "/v1/agents/knowledge/urls",
            None::<&()>,
            Some(body),
            "addKnowledgeUrl",
        )
        .await
    }
    /// One page, and when it was last read
    ///
    /// `GET /v1/agents/knowledge/urls/{id}` (`getKnowledgeUrl`).
    pub async fn get_knowledge_url(&self, id: &str) -> Result<types::KnowledgeUrl> {
        self.send(
            Method::GET,
            &format!("/v1/agents/knowledge/urls/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getKnowledgeUrl",
        )
        .await
    }
    /// Stop filling a knowledge base from a page
    ///
    /// `DELETE /v1/agents/knowledge/urls/{id}` (`deleteKnowledgeUrl`).
    pub async fn delete_knowledge_url(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/agents/knowledge/urls/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "deleteKnowledgeUrl",
        )
        .await
    }
    /// Read a page again
    ///
    /// `POST /v1/agents/knowledge/urls/{id}/index` (`indexKnowledgeUrl`).
    pub async fn index_knowledge_url(&self, id: &str) -> Result<types::KnowledgeUrl> {
        self.send(
            Method::POST,
            &format!("/v1/agents/knowledge/urls/{id}/index", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "indexKnowledgeUrl",
        )
        .await
    }
    /// What a page was last read into, in order
    ///
    /// `GET /v1/agents/knowledge/urls/{id}/passages` (`listKnowledgeUrlPassages`).
    pub async fn list_knowledge_url_passages(
        &self,
        id: &str,
    ) -> Result<Vec<types::KnowledgePassage>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/knowledge/urls/{id}/passages", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "listKnowledgeUrlPassages",
        )
        .await
    }
    /// Latest structured agent logs, with backward cursor pagination
    ///
    /// `GET /v1/agents/logs` (`listAgentLogs`).
    pub async fn list_agent_logs(&self, query: &ListAgentLogsQuery) -> Result<types::AgentLogPage> {
        self.send(
            Method::GET,
            "/v1/agents/logs",
            Some(query),
            None::<&()>,
            "listAgentLogs",
        )
        .await
    }
    /// Redacted structured log details scoped to the customer
    ///
    /// `GET /v1/agents/logs/{id}` (`getAgentLog`).
    pub async fn get_agent_log(&self, id: &str) -> Result<types::AgentLog> {
        self.send(
            Method::GET,
            &format!("/v1/agents/logs/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getAgentLog",
        )
        .await
    }
    /// The hosted MCP servers an agent may attach
    ///
    /// `GET /v1/agents/plugins` (`listPlugins`).
    pub async fn list_plugins(&self, query: &ListPluginsQuery) -> Result<Vec<types::Plugin>> {
        self.send(
            Method::GET,
            "/v1/agents/plugins",
            Some(query),
            None::<&()>,
            "listPlugins",
        )
        .await
    }
    /// Finish a plugin login
    ///
    /// `GET /v1/agents/plugins/callback` (`pluginOAuthCallback`).
    pub async fn plugin_o_auth_callback(&self, query: &PluginOAuthCallbackQuery) -> Result<()> {
        self.send(
            Method::GET,
            "/v1/agents/plugins/callback",
            Some(query),
            None::<&()>,
            "pluginOAuthCallback",
        )
        .await
    }
    /// The sessions the calling customer is running
    ///
    /// `GET /v1/agents/sessions` (`listSessions`).
    pub async fn list_sessions(&self, query: &ListSessionsQuery) -> Result<Vec<types::Session>> {
        self.send(
            Method::GET,
            "/v1/agents/sessions",
            Some(query),
            None::<&()>,
            "listSessions",
        )
        .await
    }
    /// Join a call as a voice agent
    ///
    /// `POST /v1/agents/sessions` (`createSession`).
    pub async fn create_session(
        &self,
        body: &types::CreateSessionRequest,
    ) -> Result<types::Session> {
        self.send(
            Method::POST,
            "/v1/agents/sessions",
            None::<&()>,
            Some(body),
            "createSession",
        )
        .await
    }
    /// Find a conversation by what it was called
    ///
    /// `GET /v1/agents/sessions/search` (`searchSessions`).
    pub async fn search_sessions(
        &self,
        query: &SearchSessionsQuery,
    ) -> Result<Vec<types::Session>> {
        self.send(
            Method::GET,
            "/v1/agents/sessions/search",
            Some(query),
            None::<&()>,
            "searchSessions",
        )
        .await
    }
    /// One session
    ///
    /// `GET /v1/agents/sessions/{id}` (`getSession`).
    pub async fn get_session(&self, id: &str) -> Result<types::Session> {
        self.send(
            Method::GET,
            &format!("/v1/agents/sessions/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getSession",
        )
        .await
    }
    /// Leave the call and end the session
    ///
    /// `DELETE /v1/agents/sessions/{id}` (`closeSession`).
    pub async fn close_session(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/agents/sessions/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "closeSession",
        )
        .await
    }
    /// What is known about one durable command
    ///
    /// `GET /v1/agents/sessions/{id}/commands/{command_id}` (`getSessionCommand`).
    pub async fn get_session_command(
        &self,
        id: &str,
        command_id: &str,
    ) -> Result<types::CommandReceipt> {
        self.send(
            Method::GET,
            &format!(
                "/v1/agents/sessions/{id}/commands/{command_id}",
                id = segment(id),
                command_id = segment(command_id)
            ),
            None::<&()>,
            None::<&()>,
            "getSessionCommand",
        )
        .await
    }
    /// Stop one named command, and nothing else
    ///
    /// `POST /v1/agents/sessions/{id}/commands/{command_id}/interrupt` (`interruptSessionCommand`).
    pub async fn interrupt_session_command(
        &self,
        id: &str,
        command_id: &str,
    ) -> Result<types::CommandReceipt> {
        self.send(
            Method::POST,
            &format!(
                "/v1/agents/sessions/{id}/commands/{command_id}/interrupt",
                id = segment(id),
                command_id = segment(command_id)
            ),
            None::<&()>,
            None::<&()>,
            "interruptSessionCommand",
        )
        .await
    }
    /// Continue a conversation as a new one
    ///
    /// `POST /v1/agents/sessions/{id}/fork` (`forkSession`).
    pub async fn fork_session(
        &self,
        id: &str,
        body: Option<&types::ForkSessionRequest>,
    ) -> Result<types::Session> {
        self.send(
            Method::POST,
            &format!("/v1/agents/sessions/{id}/fork", id = segment(id)),
            None::<&()>,
            body,
            "forkSession",
        )
        .await
    }
    /// Change what the agent is told to be
    ///
    /// `PUT /v1/agents/sessions/{id}/instructions` (`setSessionInstructions`).
    pub async fn set_session_instructions(
        &self,
        id: &str,
        body: &types::InstructionsRequest,
    ) -> Result<()> {
        self.send(
            Method::PUT,
            &format!("/v1/agents/sessions/{id}/instructions", id = segment(id)),
            None::<&()>,
            Some(body),
            "setSessionInstructions",
        )
        .await
    }
    /// Abandon the reply being spoken
    ///
    /// `POST /v1/agents/sessions/{id}/interrupt` (`interruptSession`).
    pub async fn interrupt_session(&self, id: &str) -> Result<()> {
        self.send(
            Method::POST,
            &format!("/v1/agents/sessions/{id}/interrupt", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "interruptSession",
        )
        .await
    }
    /// Answer a piece of text through the model, as though it had been said
    ///
    /// `POST /v1/agents/sessions/{id}/respond` (`respondSession`).
    pub async fn respond_session(
        &self,
        id: &str,
        body: &types::RespondRequest,
    ) -> Result<Option<types::CommandReceipt>> {
        self.send(
            Method::POST,
            &format!("/v1/agents/sessions/{id}/respond", id = segment(id)),
            None::<&()>,
            Some(body),
            "respondSession",
        )
        .await
    }
    /// The turns the agent took in a session
    ///
    /// `GET /v1/agents/sessions/{id}/responses` (`listResponses`).
    pub async fn list_responses(
        &self,
        id: &str,
        query: &ListResponsesQuery,
    ) -> Result<Vec<types::AgentResponse>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/sessions/{id}/responses", id = segment(id)),
            Some(query),
            None::<&()>,
            "listResponses",
        )
        .await
    }
    /// Ask the agent something and get a handle on the answer
    ///
    /// `POST /v1/agents/sessions/{id}/responses` (`createResponse`).
    pub async fn create_response(
        &self,
        id: &str,
        body: &types::CreateResponseRequest,
    ) -> Result<types::AgentResponse> {
        self.send(
            Method::POST,
            &format!("/v1/agents/sessions/{id}/responses", id = segment(id)),
            None::<&()>,
            Some(body),
            "createResponse",
        )
        .await
    }
    /// What the agent did, turn by turn, in the order it happened
    ///
    /// `GET /v1/agents/sessions/{id}/responses/items` (`listResponseItems`).
    pub async fn list_response_items(
        &self,
        id: &str,
        query: &ListResponseItemsQuery,
    ) -> Result<Vec<types::AgentResponseItem>> {
        self.send(
            Method::GET,
            &format!("/v1/agents/sessions/{id}/responses/items", id = segment(id)),
            Some(query),
            None::<&()>,
            "listResponseItems",
        )
        .await
    }
    /// Go back to a response and carry on from there
    ///
    /// `POST /v1/agents/sessions/{id}/rewind` (`rewindSession`).
    pub async fn rewind_session(&self, id: &str, body: &types::RewindSessionRequest) -> Result<()> {
        self.send(
            Method::POST,
            &format!("/v1/agents/sessions/{id}/rewind", id = segment(id)),
            None::<&()>,
            Some(body),
            "rewindSession",
        )
        .await
    }
    /// Speak a piece of text without going through the model
    ///
    /// `POST /v1/agents/sessions/{id}/say` (`saySession`).
    pub async fn say_session(&self, id: &str, body: &types::SayRequest) -> Result<()> {
        self.send(
            Method::POST,
            &format!("/v1/agents/sessions/{id}/say", id = segment(id)),
            None::<&()>,
            Some(body),
            "saySession",
        )
        .await
    }
    /// Change the models and voice of one running session
    ///
    /// `PATCH /v1/agents/sessions/{id}/settings` (`setSessionSettings`).
    pub async fn set_session_settings(
        &self,
        id: &str,
        body: &types::SessionSettingsRequest,
    ) -> Result<types::Session> {
        self.send(
            Method::PATCH,
            &format!("/v1/agents/sessions/{id}/settings", id = segment(id)),
            None::<&()>,
            Some(body),
            "setSessionSettings",
        )
        .await
    }
    /// What the simulations have come to, newest first
    ///
    /// `GET /v1/agents/simulation-runs` (`listSimulationRuns`).
    pub async fn list_simulation_runs(
        &self,
        query: &ListSimulationRunsQuery,
    ) -> Result<Vec<types::SimulationRun>> {
        self.send(
            Method::GET,
            "/v1/agents/simulation-runs",
            Some(query),
            None::<&()>,
            "listSimulationRuns",
        )
        .await
    }
    /// One run, with the conversations it had
    ///
    /// `GET /v1/agents/simulation-runs/{id}` (`getSimulationRun`).
    pub async fn get_simulation_run(&self, id: &str) -> Result<types::SimulationRun> {
        self.send(
            Method::GET,
            &format!("/v1/agents/simulation-runs/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getSimulationRun",
        )
        .await
    }
    /// Stop a run
    ///
    /// `POST /v1/agents/simulation-runs/{id}/cancel` (`cancelSimulationRun`).
    pub async fn cancel_simulation_run(&self, id: &str) -> Result<types::SimulationRun> {
        self.send(
            Method::POST,
            &format!("/v1/agents/simulation-runs/{id}/cancel", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "cancelSimulationRun",
        )
        .await
    }
    /// The simulations the calling customer has
    ///
    /// `GET /v1/agents/simulations` (`listSimulations`).
    pub async fn list_simulations(&self) -> Result<Vec<types::Simulation>> {
        self.send(
            Method::GET,
            "/v1/agents/simulations",
            None::<&()>,
            None::<&()>,
            "listSimulations",
        )
        .await
    }
    /// Write down a conversation to have with an agent
    ///
    /// `POST /v1/agents/simulations` (`createSimulation`).
    pub async fn create_simulation(
        &self,
        body: &types::SimulationRequest,
    ) -> Result<types::Simulation> {
        self.send(
            Method::POST,
            "/v1/agents/simulations",
            None::<&()>,
            Some(body),
            "createSimulation",
        )
        .await
    }
    /// One simulation
    ///
    /// `GET /v1/agents/simulations/{id}` (`getSimulation`).
    pub async fn get_simulation(&self, id: &str) -> Result<types::Simulation> {
        self.send(
            Method::GET,
            &format!("/v1/agents/simulations/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getSimulation",
        )
        .await
    }
    /// Replace a simulation
    ///
    /// `PUT /v1/agents/simulations/{id}` (`updateSimulation`).
    pub async fn update_simulation(
        &self,
        id: &str,
        body: &types::SimulationRequest,
    ) -> Result<types::Simulation> {
        self.send(
            Method::PUT,
            &format!("/v1/agents/simulations/{id}", id = segment(id)),
            None::<&()>,
            Some(body),
            "updateSimulation",
        )
        .await
    }
    /// Delete a simulation
    ///
    /// `DELETE /v1/agents/simulations/{id}` (`deleteSimulation`).
    pub async fn delete_simulation(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/agents/simulations/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "deleteSimulation",
        )
        .await
    }
    /// Have the conversations
    ///
    /// `POST /v1/agents/simulations/{id}/run` (`runSimulation`).
    pub async fn run_simulation(&self, id: &str) -> Result<types::SimulationRun> {
        self.send(
            Method::POST,
            &format!("/v1/agents/simulations/{id}/run", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "runSimulation",
        )
        .await
    }
    /// The skills the calling customer has defined
    ///
    /// `GET /v1/agents/skills` (`listSkills`).
    pub async fn list_skills(&self, query: &ListSkillsQuery) -> Result<Vec<types::Skill>> {
        self.send(
            Method::GET,
            "/v1/agents/skills",
            Some(query),
            None::<&()>,
            "listSkills",
        )
        .await
    }
    /// Define a kind of work worth handing to the slower model
    ///
    /// `POST /v1/agents/skills` (`createSkill`).
    pub async fn create_skill(&self, body: &types::SkillRequest) -> Result<types::Skill> {
        self.send(
            Method::POST,
            "/v1/agents/skills",
            None::<&()>,
            Some(body),
            "createSkill",
        )
        .await
    }
    /// One skill
    ///
    /// `GET /v1/agents/skills/{id}` (`getSkill`).
    pub async fn get_skill(&self, id: &str) -> Result<types::Skill> {
        self.send(
            Method::GET,
            &format!("/v1/agents/skills/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getSkill",
        )
        .await
    }
    /// Replace a skill
    ///
    /// `PUT /v1/agents/skills/{id}` (`updateSkill`).
    pub async fn update_skill(&self, id: &str, body: &types::SkillRequest) -> Result<types::Skill> {
        self.send(
            Method::PUT,
            &format!("/v1/agents/skills/{id}", id = segment(id)),
            None::<&()>,
            Some(body),
            "updateSkill",
        )
        .await
    }
    /// Delete a skill
    ///
    /// `DELETE /v1/agents/skills/{id}` (`deleteSkill`).
    pub async fn delete_skill(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/agents/skills/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "deleteSkill",
        )
        .await
    }
    /// Store an agent directory's instructions, skills, knowledge and settings
    ///
    /// `POST /v1/agents/sync` (`syncAgent`).
    pub async fn sync_agent(
        &self,
        body: &types::SyncAgentRequest,
    ) -> Result<types::SyncAgentResult> {
        self.send(
            Method::POST,
            "/v1/agents/sync",
            None::<&()>,
            Some(body),
            "syncAgent",
        )
        .await
    }
    /// The voices the calling customer has brought with them
    ///
    /// `GET /v1/agents/voices` (`listVoices`).
    pub async fn list_voices(&self) -> Result<Vec<types::Voice>> {
        self.send(
            Method::GET,
            "/v1/agents/voices",
            None::<&()>,
            None::<&()>,
            "listVoices",
        )
        .await
    }
    /// Name a voice of the customer's own
    ///
    /// `POST /v1/agents/voices` (`createVoice`).
    pub async fn create_voice(&self, body: &types::VoiceRequest) -> Result<types::Voice> {
        self.send(
            Method::POST,
            "/v1/agents/voices",
            None::<&()>,
            Some(body),
            "createVoice",
        )
        .await
    }
    /// The voices the speech providers offer
    ///
    /// `GET /v1/agents/voices/library` (`listLibraryVoices`).
    pub async fn list_library_voices(
        &self,
        query: &ListLibraryVoicesQuery,
    ) -> Result<types::LibraryVoices> {
        self.send(
            Method::GET,
            "/v1/agents/voices/library",
            Some(query),
            None::<&()>,
            "listLibraryVoices",
        )
        .await
    }
    /// Hear a voice from a provider's library
    ///
    /// `GET /v1/agents/voices/library/{provider}/{voice}/preview` (`previewLibraryVoice`).
    pub async fn preview_library_voice(
        &self,
        provider: &str,
        voice: &str,
    ) -> Result<types::VoicePreview> {
        self.send(
            Method::GET,
            &format!(
                "/v1/agents/voices/library/{provider}/{voice}/preview",
                provider = segment(provider),
                voice = segment(voice)
            ),
            None::<&()>,
            None::<&()>,
            "previewLibraryVoice",
        )
        .await
    }
    /// The providers a voice can be prepared with
    ///
    /// `GET /v1/agents/voices/providers` (`listVoiceProviders`).
    pub async fn list_voice_providers(&self) -> Result<types::VoiceProviders> {
        self.send(
            Method::GET,
            "/v1/agents/voices/providers",
            None::<&()>,
            None::<&()>,
            "listVoiceProviders",
        )
        .await
    }
    /// One voice, with its recordings and what each provider made of them
    ///
    /// `GET /v1/agents/voices/{id}` (`getVoice`).
    pub async fn get_voice(&self, id: &str) -> Result<types::Voice> {
        self.send(
            Method::GET,
            &format!("/v1/agents/voices/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getVoice",
        )
        .await
    }
    /// Rename a voice
    ///
    /// `PUT /v1/agents/voices/{id}` (`updateVoice`).
    pub async fn update_voice(&self, id: &str, body: &types::VoiceRequest) -> Result<types::Voice> {
        self.send(
            Method::PUT,
            &format!("/v1/agents/voices/{id}", id = segment(id)),
            None::<&()>,
            Some(body),
            "updateVoice",
        )
        .await
    }
    /// Delete a voice
    ///
    /// `DELETE /v1/agents/voices/{id}` (`deleteVoice`).
    pub async fn delete_voice(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/agents/voices/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "deleteVoice",
        )
        .await
    }
    /// Teach the text-to-speech providers this voice
    ///
    /// `POST /v1/agents/voices/{id}/prepare` (`prepareVoice`).
    pub async fn prepare_voice(
        &self,
        id: &str,
        body: &types::PrepareVoiceRequest,
    ) -> Result<types::Voice> {
        self.send(
            Method::POST,
            &format!("/v1/agents/voices/{id}/prepare", id = segment(id)),
            None::<&()>,
            Some(body),
            "prepareVoice",
        )
        .await
    }
    /// Hear a voice through one provider
    ///
    /// `POST /v1/agents/voices/{id}/preview` (`previewVoice`).
    pub async fn preview_voice(
        &self,
        id: &str,
        body: &types::VoicePreviewRequest,
    ) -> Result<types::VoicePreview> {
        self.send(
            Method::POST,
            &format!("/v1/agents/voices/{id}/preview", id = segment(id)),
            None::<&()>,
            Some(body),
            "previewVoice",
        )
        .await
    }
    /// Add a recording to a voice
    ///
    /// `POST /v1/agents/voices/{id}/samples` (`addVoiceSample`).
    pub async fn add_voice_sample(
        &self,
        id: &str,
        body: &types::VoiceSampleRequest,
    ) -> Result<types::Voice> {
        self.send(
            Method::POST,
            &format!("/v1/agents/voices/{id}/samples", id = segment(id)),
            None::<&()>,
            Some(body),
            "addVoiceSample",
        )
        .await
    }
    /// Place an outbound call and bridge it into a Stream call
    ///
    /// `POST /v1/phone/calls` (`placePhoneCall`).
    pub async fn place_phone_call(
        &self,
        body: &types::PlaceCallRequest,
    ) -> Result<types::PlacedCall> {
        self.send(
            Method::POST,
            "/v1/phone/calls",
            None::<&()>,
            Some(body),
            "placePhoneCall",
        )
        .await
    }
    /// Bring a human onto a call that is already happening
    ///
    /// `POST /v1/phone/calls/transfer` (`transferPhoneCall`).
    pub async fn transfer_phone_call(
        &self,
        body: &types::TransferCallRequest,
    ) -> Result<types::PlacedCall> {
        self.send(
            Method::POST,
            "/v1/phone/calls/transfer",
            None::<&()>,
            Some(body),
            "transferPhoneCall",
        )
        .await
    }
    /// Press digits on a call placed from here
    ///
    /// `POST /v1/phone/calls/{vendor_call_id}/digits` (`pressPhoneDigits`).
    pub async fn press_phone_digits(
        &self,
        vendor_call_id: &str,
        body: &types::PressDigitsRequest,
    ) -> Result<()> {
        self.send(
            Method::POST,
            &format!(
                "/v1/phone/calls/{vendor_call_id}/digits",
                vendor_call_id = segment(vendor_call_id)
            ),
            None::<&()>,
            Some(body),
            "pressPhoneDigits",
        )
        .await
    }
    /// The numbers the calling customer holds
    ///
    /// `GET /v1/phone/numbers` (`listPhoneNumbers`).
    pub async fn list_phone_numbers(
        &self,
        query: &ListPhoneNumbersQuery,
    ) -> Result<Vec<types::PhoneNumber>> {
        self.send(
            Method::GET,
            "/v1/phone/numbers",
            Some(query),
            None::<&()>,
            "listPhoneNumbers",
        )
        .await
    }
    /// Buy a number, which starts its monthly charge
    ///
    /// `POST /v1/phone/numbers` (`buyPhoneNumber`).
    pub async fn buy_phone_number(
        &self,
        body: &types::BuyNumberRequest,
    ) -> Result<types::PhoneNumber> {
        self.send(
            Method::POST,
            "/v1/phone/numbers",
            None::<&()>,
            Some(body),
            "buyPhoneNumber",
        )
        .await
    }
    /// Search for numbers to buy, at one vendor or all of them
    ///
    /// `GET /v1/phone/numbers/available` (`searchPhoneNumbers`).
    pub async fn search_phone_numbers(
        &self,
        query: &SearchPhoneNumbersQuery,
    ) -> Result<types::NumberSearchResult> {
        self.send(
            Method::GET,
            "/v1/phone/numbers/available",
            Some(query),
            None::<&()>,
            "searchPhoneNumbers",
        )
        .await
    }
    /// Give a number back, which stops its monthly charge
    ///
    /// `DELETE /v1/phone/numbers/{e164}` (`releasePhoneNumber`).
    pub async fn release_phone_number(&self, e164: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/phone/numbers/{e164}", e164 = segment(e164)),
            None::<&()>,
            None::<&()>,
            "releasePhoneNumber",
        )
        .await
    }
    /// Point a number at a Stream call
    ///
    /// `POST /v1/phone/numbers/{e164}/attach` (`attachPhoneNumber`).
    pub async fn attach_phone_number(
        &self,
        e164: &str,
        body: Option<&types::AttachNumberRequest>,
    ) -> Result<types::AttachedNumber> {
        self.send(
            Method::POST,
            &format!("/v1/phone/numbers/{e164}/attach", e164 = segment(e164)),
            None::<&()>,
            body,
            "attachPhoneNumber",
        )
        .await
    }
    /// List the telephony vendors and whether they can be used
    ///
    /// `GET /v1/phone/vendors` (`listPhoneVendors`).
    pub async fn list_phone_vendors(&self) -> Result<Vec<types::PhoneVendor>> {
        self.send(
            Method::GET,
            "/v1/phone/vendors",
            None::<&()>,
            None::<&()>,
            "listPhoneVendors",
        )
        .await
    }
    /// The router configs the calling customer holds
    ///
    /// `GET /v1/router/configs` (`listRouterConfigs`).
    pub async fn list_router_configs(&self) -> Result<Vec<types::RouterConfig>> {
        self.send(
            Method::GET,
            "/v1/router/configs",
            None::<&()>,
            None::<&()>,
            "listRouterConfigs",
        )
        .await
    }
    /// Store a named set of per-modality routing options
    ///
    /// `POST /v1/router/configs` (`createRouterConfig`).
    pub async fn create_router_config(
        &self,
        body: &types::RouterConfigRequest,
    ) -> Result<types::RouterConfig> {
        self.send(
            Method::POST,
            "/v1/router/configs",
            None::<&()>,
            Some(body),
            "createRouterConfig",
        )
        .await
    }
    /// One router config
    ///
    /// `GET /v1/router/configs/{id}` (`getRouterConfig`).
    pub async fn get_router_config(&self, id: &str) -> Result<types::RouterConfig> {
        self.send(
            Method::GET,
            &format!("/v1/router/configs/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getRouterConfig",
        )
        .await
    }
    /// Replace a router config
    ///
    /// `PUT /v1/router/configs/{id}` (`updateRouterConfig`).
    pub async fn update_router_config(
        &self,
        id: &str,
        body: &types::RouterConfigRequest,
    ) -> Result<types::RouterConfig> {
        self.send(
            Method::PUT,
            &format!("/v1/router/configs/{id}", id = segment(id)),
            None::<&()>,
            Some(body),
            "updateRouterConfig",
        )
        .await
    }
    /// Delete a router config
    ///
    /// `DELETE /v1/router/configs/{id}` (`deleteRouterConfig`).
    pub async fn delete_router_config(&self, id: &str) -> Result<()> {
        self.send(
            Method::DELETE,
            &format!("/v1/router/configs/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "deleteRouterConfig",
        )
        .await
    }
    /// Answer a question out of what is true now
    ///
    /// `POST /v1/search` (`search`).
    pub async fn search(&self, body: &types::SearchRequest) -> Result<types::SearchAnswer> {
        self.send(
            Method::POST,
            "/v1/search",
            None::<&()>,
            Some(body),
            "search",
        )
        .await
    }
    /// Who used the calling customer's agents, and how much
    ///
    /// `GET /v1/stats/activity` (`getActivity`).
    pub async fn get_activity(
        &self,
        query: &GetActivityQuery,
    ) -> Result<Vec<types::ActivityBucket>> {
        self.send(
            Method::GET,
            "/v1/stats/activity",
            Some(query),
            None::<&()>,
            "getActivity",
        )
        .await
    }
    /// Aggregate request rows into a rollup table
    ///
    /// `POST /v1/stats/rollup` (`runRollup`).
    pub async fn run_rollup(&self, body: &types::RollupRequest) -> Result<types::RollupResult> {
        self.send(
            Method::POST,
            "/v1/stats/rollup",
            None::<&()>,
            Some(body),
            "runRollup",
        )
        .await
    }
    /// What the calling customer spent, grouped
    ///
    /// `GET /v1/stats/spend` (`getSpend`).
    pub async fn get_spend(&self, query: &GetSpendQuery) -> Result<Vec<types::SpendBucket>> {
        self.send(
            Method::GET,
            "/v1/stats/spend",
            Some(query),
            None::<&()>,
            "getSpend",
        )
        .await
    }
    /// Which cost labels the calling customer's spend carries
    ///
    /// `GET /v1/stats/tags/keys` (`getTagKeys`).
    pub async fn get_tag_keys(&self, query: &GetTagKeysQuery) -> Result<Vec<types::TagKeySummary>> {
        self.send(
            Method::GET,
            "/v1/stats/tags/keys",
            Some(query),
            None::<&()>,
            "getTagKeys",
        )
        .await
    }
    /// Transcribe a recording, off the live path
    ///
    /// `POST /v1/stt/recordings` (`transcribeRecording`).
    pub async fn transcribe_recording(
        &self,
        body: &types::TranscriptionRequest,
    ) -> Result<types::Transcription> {
        self.send(
            Method::POST,
            "/v1/stt/recordings",
            None::<&()>,
            Some(body),
            "transcribeRecording",
        )
        .await
    }
    /// One transcription job, and its transcript once it has one
    ///
    /// `GET /v1/stt/recordings/{id}` (`getTranscription`).
    pub async fn get_transcription(&self, id: &str) -> Result<types::Transcription> {
        self.send(
            Method::GET,
            &format!("/v1/stt/recordings/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getTranscription",
        )
        .await
    }
    /// Speak a whole text into one audio file, off the live path
    ///
    /// `POST /v1/tts/recordings` (`recordSpeech`).
    pub async fn record_speech(&self, body: &types::SpeechRequest) -> Result<types::Speech> {
        self.send(
            Method::POST,
            "/v1/tts/recordings",
            None::<&()>,
            Some(body),
            "recordSpeech",
        )
        .await
    }
    /// One speech job, and its audio once it has some
    ///
    /// `GET /v1/tts/recordings/{id}` (`getSpeech`).
    pub async fn get_speech(&self, id: &str) -> Result<types::Speech> {
        self.send(
            Method::GET,
            &format!("/v1/tts/recordings/{id}", id = segment(id)),
            None::<&()>,
            None::<&()>,
            "getSpeech",
        )
        .await
    }
    /// Conversational latency for the calling customer
    ///
    /// `GET /v1/turns/stats` (`getTurnStats`).
    pub async fn get_turn_stats(
        &self,
        query: &GetTurnStatsQuery,
    ) -> Result<Vec<types::TurnStatsBucket>> {
        self.send(
            Method::GET,
            "/v1/turns/stats",
            Some(query),
            None::<&()>,
            "getTurnStats",
        )
        .await
    }
    /// List the providers configured for a modality and their live health
    ///
    /// `GET /v1/{modality}/providers` (`listProviders`).
    pub async fn list_providers(&self, modality: &str) -> Result<Vec<types::Provider>> {
        self.send(
            Method::GET,
            &format!("/v1/{modality}/providers", modality = segment(modality)),
            None::<&()>,
            None::<&()>,
            "listProviders",
        )
        .await
    }
    /// List the capability shortcuts offered as a choice, each with the models it resolves to
    ///
    /// `GET /v1/{modality}/routes` (`listRoutes`).
    pub async fn list_routes(&self, modality: &str) -> Result<Vec<types::Route>> {
        self.send(
            Method::GET,
            &format!("/v1/{modality}/routes", modality = segment(modality)),
            None::<&()>,
            None::<&()>,
            "listRoutes",
        )
        .await
    }
    /// Resolve a provider name or capability shortcut to a ranked candidate list
    ///
    /// `GET /v1/{modality}/routes/{target}` (`resolveTarget`).
    pub async fn resolve_target(
        &self,
        modality: &str,
        target: &str,
        query: &ResolveTargetQuery,
    ) -> Result<Vec<types::Candidate>> {
        self.send(
            Method::GET,
            &format!(
                "/v1/{modality}/routes/{target}",
                modality = segment(modality),
                target = segment(target)
            ),
            Some(query),
            None::<&()>,
            "resolveTarget",
        )
        .await
    }
    /// Aggregated usage for the calling customer
    ///
    /// `GET /v1/{modality}/stats` (`getStats`).
    pub async fn get_stats(
        &self,
        modality: &str,
        query: &GetStatsQuery,
    ) -> Result<Vec<types::StatsBucket>> {
        self.send(
            Method::GET,
            &format!("/v1/{modality}/stats", modality = segment(modality)),
            Some(query),
            None::<&()>,
            "getStats",
        )
        .await
    }
    /// Aggregated usage broken down by the values of one cost label
    ///
    /// `GET /v1/{modality}/stats/tags` (`getTagStats`).
    pub async fn get_tag_stats(
        &self,
        modality: &str,
        query: &GetTagStatsQuery,
    ) -> Result<Vec<types::TagStatsBucket>> {
        self.send(
            Method::GET,
            &format!("/v1/{modality}/stats/tags", modality = segment(modality)),
            Some(query),
            None::<&()>,
            "getTagStats",
        )
        .await
    }
}
/// The query string `listCalls` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListCallsQuery {
    /// Narrow to one agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Narrow to the calls one campaign placed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub campaign_id: Option<String>,
    /// Only calls that have not ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub running: Option<bool>,
    /// Only calls that started at or after this, inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Only calls that started before this, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
}
/// The query string `getCallEvents` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetCallEventsQuery {
    /// How many to return, oldest first.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
}
/// The query string `listAgentConfigs` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListAgentConfigsQuery {
    /// Narrow the list to the config with this name, which is how a name is resolved to a config. Names are unique per customer, so this answers with at most one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
/// The query string `getConversationCommand` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetConversationCommandQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}
/// The query string `getConversationMessages` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetConversationMessagesQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
}
/// The query string `listKnowledgeDocuments` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListKnowledgeDocumentsQuery {
    /// One knowledge base. Omit to list every document the customer has.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}
/// The query string `listKnowledgeUrls` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListKnowledgeUrlsQuery {
    /// One knowledge base. Omit to list every page the customer has.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
}
/// The query string `listAgentLogs` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListAgentLogsQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>,
    /// Comma-separated user/agent/tool/system sources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
}
/// The query string `listPlugins` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListPluginsQuery {
    /// Filter by name, category or description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<String>,
}
/// The query string `pluginOAuthCallback` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct PluginOAuthCallbackQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
/// The query string `listSessions` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListSessionsQuery {
    /// The agent name the session was opened against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
    /// Whose sessions to list. Only a server-side caller may set it: an end user is narrowed to their own whatever they ask for, because a filter a caller could widen is not a boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// Omitted is both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Match sessions whose custom object contains every one of these pairs, as a JSON object. Containment rather than equality, so a session carrying three labels is found by any two of them. A value that will not parse matches nothing rather than failing the request: it arrives off a query string, and one bad label should not break a conversation list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_after: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_before: Option<String>,
    /// Up to 200. Omitted is 25.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<i64>,
}
/// The query string `searchSessions` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SearchSessionsQuery {
    /// What to search for. Quoted phrases and bare words both work, and punctuation is taken rather than refused: this comes from a search box, so an apostrophe must not become a syntax error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q: Option<String>,
    /// The agent name the session was opened against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
    /// Whose sessions to list. Only a server-side caller may set it: an end user is narrowed to their own whatever they ask for, because a filter a caller could widen is not a boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// Omitted is both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Match sessions whose custom object contains every one of these pairs, as a JSON object. Containment rather than equality, so a session carrying three labels is found by any two of them. A value that will not parse matches nothing rather than failing the request: it arrives off a query string, and one bad label should not break a conversation list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_after: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_before: Option<String>,
    /// Up to 200. Omitted is 25.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<i64>,
}
/// The query string `listResponses` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListResponsesQuery {
    /// Up to 200. Omitted is 25.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<i64>,
}
/// The query string `listResponseItems` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListResponseItemsQuery {
    /// Narrow to one turn's items. Omitted is every turn in the session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// Up to 1000. Omitted is 200.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<i64>,
}
/// The query string `listSimulationRuns` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListSimulationRunsQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub simulation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
}
/// The query string `listSkills` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListSkillsQuery {
    /// Only the skills belonging to this agent config. Omit for every skill the customer has, across all of their agents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
}
/// The query string `listLibraryVoices` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListLibraryVoicesQuery {
    /// Only this provider's voices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}
/// The query string `listPhoneNumbers` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ListPhoneNumbersQuery {
    /// Include numbers that have been given back. A released number keeps its row, because what it cost while it was held is still part of that month's bill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_released: Option<bool>,
}
/// The query string `searchPhoneNumbers` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SearchPhoneNumbersQuery {
    /// One vendor to search. Absent searches every usable vendor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
    /// ISO 3166-1 alpha-2 country code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area_code: Option<String>,
    /// Digits the number must contain, anywhere in it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contains: Option<String>,
    /// Digits the number must start with, matched after the country dial code. This differs from `contains` in where the digits have to fall.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<String>,
    /// A city, region or rate centre.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locality: Option<String>,
    /// A US state or Canadian province.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub administrative_area: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub number_type: Option<String>,
    /// Capabilities every number must have. Repeat the parameter to require several. A vendor that cannot filter on one still reports what its numbers carry, so these are checked on the results either way.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
}
/// The query string `getActivity` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetActivityQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,
    /// Start of the window, inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// End of the window, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}
/// The query string `getSpend` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetSpendQuery {
    /// "modality", or the cost label to group by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,
    /// Start of the window, inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// End of the window, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    /// How many values keep a series of their own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
    /// Only count requests carrying every one of these cost labels, each written "key:value". Repeat for several.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}
/// The query string `getTagKeys` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetTagKeysQuery {
    /// Start of the window, inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// End of the window, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    /// Only consider requests carrying every one of these cost labels, each written "key:value". Repeat for several.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}
/// The query string `getTurnStats` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetTurnStatsQuery {
    /// Narrow to one agent. Omit for every agent the customer runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,
    /// Start of the window, inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// End of the window, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}
/// The query string `resolveTarget` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ResolveTargetQuery {
    /// Language hints that candidates must cover. Repeat for several.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}
/// The query string `getStats` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetStatsQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,
    /// Start of the window, inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// End of the window, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    /// Only count requests carrying every one of these cost labels, each written "key:value". Repeat for several. Filtering reads the request rows rather than the rollups, since a rollup bucket no longer knows which labels its requests carried.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}
/// The query string `getTagStats` takes. A field left `None` is left out.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct GetTagStatsQuery {
    /// The cost label to group by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granularity: Option<String>,
    /// Start of the window, inclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// End of the window, exclusive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}
