use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use serde_json::{Map, Value};
use tokio::sync::Mutex;

use crate::client::Client;
use crate::dispatch::{Dispatch, InboundCall, InboundMessage};
use crate::error::{Error, Result};
use crate::folder::{self, Folder};
use crate::harness::Harness;
use crate::knowledge::Knowledge;
use crate::operations::ListAgentConfigsQuery;
use crate::session::{Session, WatchOptions};
use crate::stream::StreamApp;
use crate::tools::Tools;
use crate::types;

/// The memory filter key naming who the memories are about. Everything else narrows recall.
pub const USER_KEY: &str = "user_id";

/// A configured agent, before and between the calls it holds.
///
/// An agent here is configuration and function calling. The conversation itself — joining
/// the call, hearing the caller, answering and speaking — happens in the backend, and what
/// arrives here are the events saying so.
///
/// ```no_run
/// # async fn example() -> vision_agents::Result<()> {
/// use vision_agents::Agent;
///
/// let agent = Agent::new("simple_voice_ai")
///     .cost_tracking([("env", "production")])
///     .memory_filter([("user_id", "123")]);
/// let session = agent.join().await?;
/// session.wait().await;
/// # Ok(())
/// # }
/// ```
pub struct Agent {
    name: String,
    config: Option<String>,
    instructions: String,
    guardrail: String,
    folder: Option<Folder>,
    harness: Option<Harness>,
    pipeline: types::CreateSessionRequest,
    cost_tracking: BTreeMap<String, String>,
    memory_filter: BTreeMap<String, String>,
    user_id: String,
    tools: Tools,
    watch: WatchOptions,
    client: OnceLock<Client>,
    stream: OnceLock<StreamApp>,
    /// Whether the agent's directory has been looked for and stored, which happens once.
    ensured: Mutex<bool>,
}

impl Agent {
    /// An agent run from the config stored under `config`.
    ///
    /// If a directory of that name is found (`examples/*/<config>`, `agents/<config>` or
    /// `<config>`, from the working directory up), it is stored before the first session
    /// opens, and only when it changed since `.agent_sync` was written. Anything set here as
    /// well wins over what the config says, for these sessions only.
    pub fn new(config: impl Into<String>) -> Self {
        let config = config.into();
        let mut agent = Agent::named(config.clone());
        agent.config = Some(config);
        agent
    }

    /// An agent read from a directory, stored under the name its agent.yaml gives it.
    pub fn from_folder(path: impl AsRef<Path>) -> Result<Self> {
        let folder = Folder::load(path)?;
        let mut agent = Agent::new(folder.name.clone());
        agent.fold(folder);
        Ok(agent)
    }

    /// An agent spelled out in code rather than stored: each session carries its whole
    /// configuration. [`Agent::sync`] stores it, after which [`Agent::new`] can name it.
    pub fn named(name: impl Into<String>) -> Self {
        let name = name.into();
        Agent {
            user_id: user_id_of(&name),
            name,
            config: None,
            instructions: String::new(),
            guardrail: String::new(),
            folder: None,
            harness: None,
            pipeline: types::CreateSessionRequest::default(),
            cost_tracking: BTreeMap::new(),
            memory_filter: BTreeMap::new(),
            tools: Tools::new(),
            watch: WatchOptions::default(),
            client: OnceLock::new(),
            stream: OnceLock::new(),
            ensured: Mutex::new(false),
        }
    }

    /// The system prompt. It reaches the backend with [`Agent::sync`]: a session runs on the
    /// stored config's.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// A guardrail.md: frontmatter saying how a turn is screened, then the policy in prose.
    /// It is enforced in the backend, so a turn the policy refuses never reaches the model.
    pub fn guardrail(mut self, guardrail: impl Into<String>) -> Self {
        self.guardrail = guardrail.into();
        self
    }

    /// Which models hold the conversation, and how: the pipeline fields of a session
    /// (`llm`, `stt`, `tts`, `sts`, `voice`, `greeting`, `languages`, `video` and the rest).
    /// What is left `None` is left to the config, or to the backend's defaults.
    pub fn pipeline(mut self, pipeline: types::CreateSessionRequest) -> Self {
        self.pipeline = pipeline;
        self
    }

    /// The model that answers: a provider/model name or a shortcut such as `llm-fast`.
    pub fn llm(mut self, llm: impl Into<String>) -> Self {
        self.pipeline.llm = Some(llm.into());
        self
    }

    /// What stands between the caller and the model: skills, subagents and the sandbox.
    /// Stored on the agent's config by [`Agent::sync`], never sent with a session.
    pub fn harness(mut self, harness: Harness) -> Self {
        self.harness = Some(harness);
        self
    }

    /// Labels every request the session makes, so spend can be attributed.
    pub fn cost_tracking<K: Into<String>, V: Into<String>>(
        mut self,
        tags: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.cost_tracking = tags
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }

    /// Who the memories are about, under `user_id`, and what else narrows recall.
    pub fn memory_filter<K: Into<String>, V: Into<String>>(
        mut self,
        filter: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        self.memory_filter = filter
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }

    /// Who the agent joins a call as. Derived from the name when not set.
    pub fn user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = user_id.into();
        self
    }

    /// The caller's own functions, which the model is offered and this process runs.
    pub fn tools(mut self, tools: Tools) -> Self {
        self.tools = tools;
        self
    }

    /// How the session's events socket is opened.
    pub fn watch(mut self, watch: WatchOptions) -> Self {
        self.watch = watch;
        self
    }

    /// The router this agent talks to. Built from the environment when not set.
    pub fn client(self, client: Client) -> Self {
        let _ = self.client.set(client);
        self
    }

    /// Signs the links to a session's call. Built from the environment when needed.
    pub fn stream(self, stream: StreamApp) -> Self {
        let _ = self.stream.set(stream);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The directory this agent was read from, once it has been found.
    pub fn folder(&self) -> Option<&Folder> {
        self.folder.as_ref()
    }

    /// The functions this agent runs for the model.
    pub fn functions(&self) -> &Tools {
        &self.tools
    }

    /// The router this agent talks to.
    pub fn router(&self) -> Result<&Client> {
        if let Some(client) = self.client.get() {
            return Ok(client);
        }
        let client = Client::from_env()?;
        Ok(self.client.get_or_init(|| client))
    }

    /// Where more knowledge goes: the knowledge base of the config this agent runs from.
    pub fn knowledge(&self) -> Result<Knowledge> {
        let Some(config) = &self.config else {
            return Err(Error::configuration(
                "a knowledge base belongs to a stored config; build the agent with Agent::new(config)",
            ));
        };
        Ok(Knowledge::new(self.router()?.clone(), config.clone()))
    }

    /// Holds a conversation on the session's own call, `agent:<session id>`.
    ///
    /// It returns once the backend is in the call, so an agent that has joined is already
    /// listening.
    pub async fn join(&self) -> Result<Session> {
        self.join_with(types::CreateSessionRequest::default()).await
    }

    /// [`Agent::join`], with whatever else the session is opened with. What is set in
    /// `request` wins over what the agent would have said.
    pub async fn join_with(&self, mut request: types::CreateSessionRequest) -> Result<Session> {
        request.start_voice = Some(true);
        self.open(request).await
    }

    /// A link a person can open to join this session's call from a browser and hear it.
    /// The session needs voice started.
    pub fn monitor_url(&self, session: &Session) -> Result<String> {
        if !session.voice_started() {
            return Err(Error::configuration(
                "a conversation held in writing has no call to watch",
            ));
        }
        self.stream_app()?.monitor_url(
            &session.call(),
            &format!("monitor-{}", session.id()),
            "Monitor",
        )
    }

    /// Holds the conversation in writing rather than on a call.
    pub async fn chat(&self) -> Result<Session> {
        self.chat_with(types::CreateSessionRequest::default()).await
    }

    /// [`Agent::chat`], with whatever else the session is opened with: a title, `incognito`
    /// and the rest. What is set in `request` wins over what the agent would have said.
    pub async fn chat_with(&self, mut request: types::CreateSessionRequest) -> Result<Session> {
        request.start_voice = None;
        self.open(request).await
    }

    /// Carries on a conversation held in writing, by the id of the session it was held in.
    /// One that ended is reopened with what was said in it.
    pub async fn resume(&self, id: &str) -> Result<Session> {
        let client = self.router()?;
        let got = client.get_session(id).await?;
        Session::watching(client, got, self.tools.clone(), self.watch).await
    }

    /// Answers a call that arrived on the dispatch socket, on the call the router routed the
    /// caller into, which is named for the session opened here. The number they reached is
    /// what the agent acts from.
    pub async fn answer(&self, call: &InboundCall) -> Result<Session> {
        if call.session_id.is_empty() {
            return Err(Error::configuration(
                "the call names no session; attach its number again",
            ));
        }
        self.open(types::CreateSessionRequest {
            id: Some(call.session_id.clone()),
            start_voice: Some(true),
            phone: (!call.called_number.is_empty()).then(|| types::SessionPhone {
                number: call.called_number.clone(),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
    }

    /// Answers a message written to an agent that is not running, in the channel its agent
    /// id names, so whoever wrote it is already reading the answer as it is generated.
    pub async fn reply(&self, message: &InboundMessage) -> Result<Session> {
        self.chat_with(types::CreateSessionRequest {
            agent_id: (!message.agent_id.is_empty()).then(|| message.agent_id.clone()),
            ..Default::default()
        })
        .await
    }

    /// Rings `to` from `from` and holds the conversation when they answer.
    ///
    /// The agent placed this call, so it is told it is navigating: recordings are let
    /// finish and menus are answered rather than talked over.
    pub async fn outbound_call(&self, from: &str, to: &str) -> Result<Session> {
        if from.is_empty() || to.is_empty() {
            return Err(Error::configuration(
                "a call needs a number to ring from and one to ring",
            ));
        }
        // Placing the call makes its own routing rule pinned to the call of the session it
        // names, so the answered leg arrives in the call this agent is about to join.
        let placed = self
            .router()?
            .place_phone_call(&types::PlaceCallRequest {
                from: from.into(),
                to: to.into(),
                tags: self.cost_tracking.clone(),
                ..Default::default()
            })
            .await?;
        let Some(session_id) = placed.session_id else {
            return Err(Error::Failed {
                operation: "placing the call".into(),
                message: "the router placed the call for no session".into(),
            });
        };
        self.open(types::CreateSessionRequest {
            id: Some(session_id),
            start_voice: Some(true),
            navigating: Some(true),
            phone: Some(types::SessionPhone {
                number: from.into(),
                vendor_call_id: Some(placed.vendor_call_id),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
    }

    /// Answers the next call to `number`, returning once somebody has rung and said
    /// something. For more than one call at a time, use [`crate::Dispatch`].
    ///
    /// The number is attached, so every caller lands in a call of their own, and this waits
    /// for the router to hand the next one over.
    pub async fn wait_for_call(&self, number: &str) -> Result<Session> {
        if number.is_empty() {
            return Err(Error::configuration("there is no number to answer on"));
        }
        let client = self.router()?;
        client
            .attach_phone_number(number, Some(&types::AttachNumberRequest::default()))
            .await?;

        let dispatch = Dispatch::with_capacity(client.clone(), 1);
        let (arrived, mut calls) = tokio::sync::mpsc::channel::<InboundCall>(1);
        let wanted = number.to_string();
        dispatch.wait_for_call(move |call| {
            let (arrived, wanted) = (arrived.clone(), wanted.clone());
            async move {
                if call.called_number != wanted {
                    return Err(Error::configuration(format!(
                        "this agent is waiting on {wanted}, not {}",
                        call.called_number
                    )));
                }
                arrived
                    .try_send(call)
                    .map_err(|_| Error::configuration("this agent is already answering a call"))
            }
        });
        let mut running = {
            let dispatch = dispatch.clone();
            tokio::spawn(async move { dispatch.run().await })
        };
        let call = tokio::select! {
            Some(call) = calls.recv() => call,
            ended = &mut running => {
                return Err(match ended {
                    Ok(Err(error)) => error,
                    _ => Error::Closed("dispatch, before anybody rang,".into()),
                });
            }
        };
        dispatch.stop();
        let _ = running.await;

        let session = self.answer(&call).await?;
        if session.wait_for_event("heard").await.is_some() {
            return Ok(session);
        }
        session.close().await;
        Err(Error::Closed("the call, before anybody rang,".into()))
    }

    /// Stores the agent in the backend: its instructions, guardrail, skills and knowledge,
    /// and the models it was declared with.
    ///
    /// An agent read from a directory is synced in one request carrying a fingerprint of
    /// everything in it, and `.agent_sync` records it, so a sync of a directory nothing has
    /// touched only reads the stored config back. An agent spelled out in code has its
    /// skills and config written by name, editing whatever is already stored under it.
    /// Server side only.
    pub async fn sync(&self) -> Result<types::AgentConfig> {
        if let Some(harness) = &self.harness {
            harness.validate()?;
        }
        match &self.folder {
            Some(folder) => self.sync_folder(folder).await,
            None => self.sync_config().await,
        }
    }

    /// Renders the agent's configuration into a session and opens it.
    async fn open(&self, overrides: types::CreateSessionRequest) -> Result<Session> {
        self.ensure().await?;
        let request = types::CreateSessionRequest {
            user_id: Some(self.user_id.clone()),
            user_name: Some(self.name.clone()),
            agent_id: Some(self.user_id.clone()),
            agent: self.config.clone(),
            tags: self.cost_tracking.clone(),
            memory: memory_of(&self.memory_filter),
            ..self.pipeline.clone()
        };
        Session::open(
            self.router()?,
            overlay(request, overrides)?,
            self.tools.clone(),
            self.watch,
        )
        .await
    }

    /// Stores the agent's directory before its first session, if it has one and it changed.
    async fn ensure(&self) -> Result<()> {
        let mut ensured = self.ensured.lock().await;
        if *ensured {
            return Ok(());
        }
        if let Some(folder) = &self.folder {
            if folder::read_stamp(&folder.path) != self.folder_hash(folder) {
                self.sync_folder(folder).await?;
            }
        } else if let Some(config) = &self.config {
            let found = std::env::current_dir()
                .ok()
                .and_then(|here| Folder::find(config, here));
            if let Some(path) = found {
                let folder = Folder::load(path)?;
                if folder::read_stamp(&folder.path) != self.folder_hash(&folder) {
                    self.sync_folder(&folder).await?;
                }
            }
        }
        *ensured = true;
        Ok(())
    }

    fn fold(&mut self, folder: Folder) {
        if self.instructions.is_empty() {
            self.instructions = folder.instructions.clone();
        }
        if self.guardrail.is_empty() {
            self.guardrail = folder.guardrail.clone();
        }
        if !folder.skills.is_empty() {
            match &mut self.harness {
                None => {
                    self.harness = Some(Harness {
                        skills: folder.skills.clone(),
                        ..Harness::default()
                    })
                }
                Some(harness) if harness.skills.is_empty() => {
                    harness.skills = folder.skills.clone()
                }
                Some(_) => {}
            }
        }
        self.folder = Some(folder);
    }

    /// The skills the stored config should name: the harness's, or the directory's.
    fn synced_skills<'a>(&'a self, folder: Option<&'a Folder>) -> &'a [crate::harness::Skill] {
        match &self.harness {
            Some(harness) if !harness.skills.is_empty() => &harness.skills,
            _ => folder.map_or(&[], |folder| &folder.skills),
        }
    }

    /// What the harness sets on the agent's config: its name, subagent and sandbox. `None`
    /// is left out, so the router keeps whatever is already stored for it.
    fn stored_harness(
        &self,
    ) -> (
        Option<types::Harness>,
        Option<String>,
        Option<types::Sandbox>,
    ) {
        match &self.harness {
            None => (None, None, None),
            Some(harness) => (harness.name, text(harness.subagent()), harness.vm),
        }
    }

    /// The directory's fingerprint, extended by what the code set, so changing either one
    /// syncs again. Taken the way the Go SDK takes it, so their stamps agree.
    fn folder_hash(&self, folder: &Folder) -> String {
        let instructions = if self.instructions.is_empty() {
            &folder.instructions
        } else {
            &self.instructions
        };
        let guardrail = if self.guardrail.is_empty() {
            &folder.guardrail
        } else {
            &self.guardrail
        };
        let hash = folder::fingerprint(
            &folder.declaration,
            instructions,
            guardrail,
            self.synced_skills(Some(folder)),
            &folder.knowledge,
            &folder.knowledge_urls,
            folder.simulations.as_deref(),
        );
        let (harness, subagent, sandbox) = self.stored_harness();
        if harness.is_none()
            && subagent.is_none()
            && sandbox.is_none()
            && self.cost_tracking.is_empty()
        {
            return hash;
        }
        // Go's fmt.Sprint of three strings, which runs them together, and of a map, which
        // sorts its keys.
        let stored = format!(
            "{}{}{}",
            harness.map(|name| name.to_string()).unwrap_or_default(),
            subagent.unwrap_or_default(),
            sandbox.map(|vm| vm.to_string()).unwrap_or_default(),
        );
        let tags: Vec<String> = self
            .cost_tracking
            .iter()
            .map(|(key, value)| format!("{key}:{value}"))
            .collect();
        folder::fingerprint(
            &hash,
            &stored,
            &format!("map[{}]", tags.join(" ")),
            &[],
            &[],
            &[],
            None,
        )
    }

    async fn sync_folder(&self, folder: &Folder) -> Result<types::AgentConfig> {
        let client = self.router()?;
        let name = self.config.clone().unwrap_or_else(|| self.name.clone());
        let hash = self.folder_hash(folder);
        if folder::read_stamp(&folder.path) == hash
            && let Some(stored) = stored_config(client, &name).await?
        {
            return Ok(stored);
        }

        let settings = &folder.settings;
        let instructions = if self.instructions.is_empty() {
            &folder.instructions
        } else {
            &self.instructions
        };
        let guardrail = if self.guardrail.is_empty() {
            &folder.guardrail
        } else {
            &self.guardrail
        };
        let skills = self.synced_skills(Some(folder));
        let mut tags = settings.tags.clone();
        tags.extend(self.cost_tracking.clone());
        let (harness, subagent, sandbox) = self.stored_harness();

        let body = types::SyncAgentRequest {
            name,
            hash: hash.clone(),
            instructions: text(instructions),
            guardrail: text(guardrail),
            skills: (!skills.is_empty())
                .then(|| skills.iter().map(|skill| skill.request()).collect()),
            knowledge: (!folder.knowledge.is_empty()).then(|| {
                folder
                    .knowledge
                    .iter()
                    .map(|document| types::KnowledgeDocument {
                        source: document.source.clone(),
                        text: document.text.clone(),
                    })
                    .collect()
            }),
            knowledge_urls: (!folder.knowledge_urls.is_empty()).then(|| {
                folder
                    .knowledge_urls
                    .iter()
                    .map(|page| types::KnowledgeUrlDeclaration {
                        url: page.url.clone(),
                        title: text(&page.title),
                        description: text(&page.description),
                        refresh_hours: (page.refresh_hours > 0).then_some(page.refresh_hours),
                    })
                    .collect()
            }),
            simulations: folder.simulations.as_ref().map(|simulations| {
                simulations
                    .iter()
                    .map(folder::Simulation::declaration)
                    .collect()
            }),
            mode: settings.mode,
            stt: text(&settings.stt),
            tts: text(&settings.tts),
            sts: settings.sts.clone(),
            voice: text(&settings.voice),
            llm: text(&settings.llm),
            harness: harness.or(settings.harness),
            subagent: subagent.or_else(|| text(&settings.subagent)),
            search: text(&settings.search),
            greeting: settings.greeting.clone(),
            sandbox: sandbox.or(settings.sandbox),
            plugins: (!settings.plugins.is_empty()).then(|| {
                settings
                    .plugins
                    .iter()
                    .cloned()
                    .map(types::PluginEntry::String)
                    .collect()
            }),
            keyterms: (!settings.keyterms.is_empty()).then(|| settings.keyterms.clone()),
            tags,
            video: settings.video.as_ref().map(|video| types::SessionVideo {
                max_frames: Some(video.max_frames),
                source: text(&video.source),
            }),
            dispatch: settings.dispatch.clone(),
            ..Default::default()
        };
        let synced = client.sync_agent(&body).await?;
        folder::write_stamp(&folder.path, &hash)?;
        Ok(synced.config)
    }

    async fn sync_config(&self) -> Result<types::AgentConfig> {
        let client = self.router()?;
        let skills = self.synced_skills(None);
        if !skills.is_empty() {
            let known: BTreeMap<String, String> = client
                .list_skills(&Default::default())
                .await?
                .into_iter()
                .map(|skill| (skill.name, skill.id))
                .collect();
            for skill in skills {
                match known.get(&skill.name) {
                    Some(id) => client.update_skill(id, &skill.request()).await?,
                    None => client.create_skill(&skill.request()).await?,
                };
            }
        }

        let name = self.config.clone().unwrap_or_else(|| self.name.clone());
        let (harness, subagent, sandbox) = self.stored_harness();
        let wanted = types::AgentConfigRequest {
            name: name.clone(),
            instructions: text(&self.instructions),
            guardrail: text(&self.guardrail),
            harness,
            subagent,
            sandbox,
            tags: self.cost_tracking.clone(),
            skills: (!skills.is_empty())
                .then(|| skills.iter().map(|skill| skill.name.clone()).collect()),
            ..Default::default()
        };
        match stored_config(client, &name).await? {
            Some(stored) => client.update_agent_config(&stored.id, &wanted).await,
            None => client.create_agent_config(&wanted).await,
        }
    }

    fn stream_app(&self) -> Result<&StreamApp> {
        if let Some(stream) = self.stream.get() {
            return Ok(stream);
        }
        let stream = StreamApp::from_env()?;
        Ok(self.stream.get_or_init(|| stream))
    }
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Agent")
            .field("name", &self.name)
            .field("config", &self.config)
            .finish()
    }
}

/// Turns a name into something a call can be joined under.
pub fn user_id_of(name: &str) -> String {
    let id: String = name
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let id = id.trim_matches('-');
    if id.is_empty() {
        "vision-agent".into()
    } else {
        id.into()
    }
}

/// Splits the filter into who the memories are about and what narrows them.
fn memory_of(filter: &BTreeMap<String, String>) -> Option<types::SessionMemory> {
    if filter.is_empty() {
        return None;
    }
    let mut memory = types::SessionMemory::default();
    for (key, value) in filter {
        if key == USER_KEY {
            memory.user_id = Some(value.clone());
        } else {
            memory.filter.insert(key.clone(), value.clone());
        }
    }
    Some(memory)
}

/// Writes every field `overrides` sets over `base`, the way a spread does.
fn overlay(
    base: types::CreateSessionRequest,
    overrides: types::CreateSessionRequest,
) -> Result<types::CreateSessionRequest> {
    let as_object = |request: &types::CreateSessionRequest| match serde_json::to_value(request) {
        Ok(Value::Object(object)) => Ok(object),
        _ => Err(Error::configuration("a session request is an object")),
    };
    let mut merged: Map<String, Value> = as_object(&base)?;
    merged.extend(as_object(&overrides)?);
    serde_json::from_value(Value::Object(merged)).map_err(|source| Error::Decode {
        operation: "rendering the session".into(),
        source,
    })
}

async fn stored_config(client: &Client, name: &str) -> Result<Option<types::AgentConfig>> {
    let stored = client
        .list_agent_configs(&ListAgentConfigsQuery {
            name: Some(name.into()),
        })
        .await?;
    Ok(stored.into_iter().find(|config| config.name == name))
}

fn text(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}
