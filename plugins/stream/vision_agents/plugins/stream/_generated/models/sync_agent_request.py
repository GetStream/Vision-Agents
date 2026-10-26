from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.agent_mode import AgentMode
from ..models.agent_role import AgentRole
from ..models.harness import Harness
from ..models.sandbox import Sandbox
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_channels import AgentChannels
    from ..models.agent_connector_binding import AgentConnectorBinding
    from ..models.agent_dispatch import AgentDispatch
    from ..models.agent_scheduling import AgentScheduling
    from ..models.agent_tools import AgentTools
    from ..models.greeting import Greeting
    from ..models.knowledge_document import KnowledgeDocument
    from ..models.knowledge_url_declaration import KnowledgeUrlDeclaration
    from ..models.left_out_connector import LeftOutConnector
    from ..models.mcp_server import McpServer
    from ..models.plugin_event import PluginEvent
    from ..models.plugin_with_options import PluginWithOptions
    from ..models.resume_history import ResumeHistory
    from ..models.sandbox_options import SandboxOptions
    from ..models.session_video import SessionVideo
    from ..models.simulation_declaration import SimulationDeclaration
    from ..models.skill_request import SkillRequest
    from ..models.sync_agent_request_tags import SyncAgentRequestTags


T = TypeVar("T", bound="SyncAgentRequest")


@_attrs_define
class SyncAgentRequest:
    """An agent directory as it is on disk. Everything after the simulations is what the directory's declaration decides
    rather than what it holds, and a setting left out leaves whatever is stored, so a model chosen in the dashboard
    survives a sync that says nothing about it.

        Attributes:
            hash_ (str): A fingerprint of the directory. A second sync with the same hash does nothing.
            name (str): What the config is called, which is also the directory's name.
            base_change (str | Unset): The newest change the caller has already seen, as last_change named it. Everything up
                to it is taken as decided, so the sync is not refused for it again.
            channels (AgentChannels | Unset): The lines this agent answers on besides its Stream Chat channel. Each names a
                number the app connected with POST /v1/agents/channels, and only one agent may answer on a number. A message
                that arrives is answered in the sender's own conversation, so what they say is kept and shown wherever the rest
                of it is.
            check_changes (bool | Unset): Refuse the sync, with unsynced_changes, when somebody has changed one of the
                settings it would write since the last sync -- in the dashboard, say. A client that asks for this shows the
                person what changed (GET /v1/agents/configs/{id}/changes) and syncs again with base_change once they have
                decided. Omitted, the sync writes over whatever is there, which is what a process syncing on startup wants.
            connectors (list[AgentConnectorBinding] | Unset): The connectors agent.yaml binds. Sent, they are the whole of
                the agent's bindings and replace the ones stored, an empty list removing them all. Left out, the stored ones are
                left alone.
            dispatch (AgentDispatch | Unset): What the agent leaves to the customer's own server, which waits on
                /v1/dispatch. Omitted settings are disabled.
            greeting (Greeting | Unset): What the agent says as it joins, before anyone speaks.
            guardrail (str | Unset): The directory's guardrail.md, whole: frontmatter saying how to screen a turn, then the
                policy in prose. Empty means every turn is answered.
            harness (Harness | Unset): Which harness the agent's sessions run: what hands work to the subagent, loads
                skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the
                default, the only one there is.
            instructions (str | Unset):
            keyterms (list[str] | Unset):
            knowledge (list[KnowledgeDocument] | Unset):
            knowledge_urls (list[KnowledgeUrlDeclaration] | Unset): The pages the directory's knowledge/urls.yaml declares.
                They are subscribed to in the same knowledge base as the files, so one lookup covers both.
            left_out_connectors (list[LeftOutConnector] | Unset): The connectors agent.yaml names that the client left out
                of connectors, because the app has no connection by the id or label they name. Stored as sent and served on the
                config, so the dashboard can say what the agent is missing. A non-empty list is refused without connectors: a
                sync that sends connectors replaces the stored ones with these, or with none when this is left out, and a sync
                that sends neither leaves them alone. At most 64, as connectors.
            llm (str | Unset):
            mcp_servers (list[McpServer] | Unset): MCP servers outside the plugin catalog, opened by their URL with no
                login.
            mode (AgentMode | Unset): Whether the agent is spoken to or written to. A voice agent joins a call, transcribes
                what it hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither
                speech target and a session created from it needs no call to join.
            plugin_events (list[PluginEvent] | Unset): Deprecated: use the events of a fixed binding under connectors. MCP
                events the agent subscribes to on its plugins, each opening a text conversation when it arrives.
            plugins (list[PluginWithOptions | str] | Unset): Deprecated: use connectors, a binding to a connector. Plugins
                the agent reaches: a catalog id, or an object naming it with how it is reached. The app connects each once,
                unless its entry sets user: then each end user connects it with their own account, from the conversation, the
                first time the agent needs it.
            resume_history (ResumeHistory | Unset): How a conversation is handed back to the model when a session carries it
                on.
            role (AgentRole | Unset): What the agent is for in the app, beyond answering people. agent_moderator is the
                app's moderator agent, which reviews flagged messages. One live agent per app holds a role: giving it to a
                second is refused, and a test copy (tagged draft_of) may share the role of the agent it copies. Left out on an
                update or a sync, the stored role stays. Omit it for an ordinary agent.
            sandbox (Sandbox | Unset): Where the agent may run code it writes. A text agent runs it itself; a voice agent
                hands it to the subagent, since running code takes seconds a call has none of. Omit it and the agent works
                everything out in its head.
            sandbox_options (SandboxOptions | Unset): How the sandbox is built, what it may reach and how long code may run
                in it. Only meaningful with a sandbox. Omit it for the provider's own Python sandbox, no network and a 30 second
                run.
            scheduling (AgentScheduling | Unset): Whether the agent may schedule tasks for its users, and the defaults those
                tasks take.
            search (str | Unset):
            simulations (list[SimulationDeclaration] | Unset): The simulations the directory's simulations/*.yaml declare.
                Sent, they are the whole of the agent's simulations: each is found by name, and one no longer declared is
                deleted. Left out, the stored ones are left alone.
            skills (list[SkillRequest] | Unset):
            sts (str | Unset): A speech-to-speech target: one native audio model that hears the caller and speaks back.
                Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade.
            stt (str | Unset):
            subagent (str | Unset): Only a voice agent names one: a text agent runs everything on its llm.
            tags (SyncAgentRequestTags | Unset):
            tools (AgentTools | Unset): How an agent is offered its tools, and how long what they were given and gave back
                is kept.
            tts (str | Unset):
            video (SessionVideo | Unset):
            voice (str | Unset): The voice the agent speaks in, as TtsOptions.voice takes it: a provider's own voice id, or
                one of your voices, with the custom: prefix to mean only yours. A native agent (sts) takes the vendor's own
                voice name instead.
    """

    hash_: str
    name: str
    base_change: str | Unset = UNSET
    channels: AgentChannels | Unset = UNSET
    check_changes: bool | Unset = UNSET
    connectors: list[AgentConnectorBinding] | Unset = UNSET
    dispatch: AgentDispatch | Unset = UNSET
    greeting: Greeting | Unset = UNSET
    guardrail: str | Unset = UNSET
    harness: Harness | Unset = UNSET
    instructions: str | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    knowledge: list[KnowledgeDocument] | Unset = UNSET
    knowledge_urls: list[KnowledgeUrlDeclaration] | Unset = UNSET
    left_out_connectors: list[LeftOutConnector] | Unset = UNSET
    llm: str | Unset = UNSET
    mcp_servers: list[McpServer] | Unset = UNSET
    mode: AgentMode | Unset = UNSET
    plugin_events: list[PluginEvent] | Unset = UNSET
    plugins: list[PluginWithOptions | str] | Unset = UNSET
    resume_history: ResumeHistory | Unset = UNSET
    role: AgentRole | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    sandbox_options: SandboxOptions | Unset = UNSET
    scheduling: AgentScheduling | Unset = UNSET
    search: str | Unset = UNSET
    simulations: list[SimulationDeclaration] | Unset = UNSET
    skills: list[SkillRequest] | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    tags: SyncAgentRequestTags | Unset = UNSET
    tools: AgentTools | Unset = UNSET
    tts: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.plugin_with_options import PluginWithOptions

        hash_ = self.hash_

        name = self.name

        base_change = self.base_change

        channels: dict[str, Any] | Unset = UNSET
        if not isinstance(self.channels, Unset):
            channels = self.channels.to_dict()

        check_changes = self.check_changes

        connectors: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.connectors, Unset):
            connectors = []
            for connectors_item_data in self.connectors:
                connectors_item = connectors_item_data.to_dict()
                connectors.append(connectors_item)

        dispatch: dict[str, Any] | Unset = UNSET
        if not isinstance(self.dispatch, Unset):
            dispatch = self.dispatch.to_dict()

        greeting: dict[str, Any] | Unset = UNSET
        if not isinstance(self.greeting, Unset):
            greeting = self.greeting.to_dict()

        guardrail = self.guardrail

        harness: str | Unset = UNSET
        if not isinstance(self.harness, Unset):
            harness = self.harness.value

        instructions = self.instructions

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

        knowledge: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.knowledge, Unset):
            knowledge = []
            for knowledge_item_data in self.knowledge:
                knowledge_item = knowledge_item_data.to_dict()
                knowledge.append(knowledge_item)

        knowledge_urls: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.knowledge_urls, Unset):
            knowledge_urls = []
            for knowledge_urls_item_data in self.knowledge_urls:
                knowledge_urls_item = knowledge_urls_item_data.to_dict()
                knowledge_urls.append(knowledge_urls_item)

        left_out_connectors: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.left_out_connectors, Unset):
            left_out_connectors = []
            for left_out_connectors_item_data in self.left_out_connectors:
                left_out_connectors_item = left_out_connectors_item_data.to_dict()
                left_out_connectors.append(left_out_connectors_item)

        llm = self.llm

        mcp_servers: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.mcp_servers, Unset):
            mcp_servers = []
            for mcp_servers_item_data in self.mcp_servers:
                mcp_servers_item = mcp_servers_item_data.to_dict()
                mcp_servers.append(mcp_servers_item)

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        plugin_events: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.plugin_events, Unset):
            plugin_events = []
            for plugin_events_item_data in self.plugin_events:
                plugin_events_item = plugin_events_item_data.to_dict()
                plugin_events.append(plugin_events_item)

        plugins: list[dict[str, Any] | str] | Unset = UNSET
        if not isinstance(self.plugins, Unset):
            plugins = []
            for plugins_item_data in self.plugins:
                plugins_item: dict[str, Any] | str
                if isinstance(plugins_item_data, PluginWithOptions):
                    plugins_item = plugins_item_data.to_dict()
                else:
                    plugins_item = plugins_item_data
                plugins.append(plugins_item)

        resume_history: dict[str, Any] | Unset = UNSET
        if not isinstance(self.resume_history, Unset):
            resume_history = self.resume_history.to_dict()

        role: str | Unset = UNSET
        if not isinstance(self.role, Unset):
            role = self.role.value

        sandbox: str | Unset = UNSET
        if not isinstance(self.sandbox, Unset):
            sandbox = self.sandbox.value

        sandbox_options: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sandbox_options, Unset):
            sandbox_options = self.sandbox_options.to_dict()

        scheduling: dict[str, Any] | Unset = UNSET
        if not isinstance(self.scheduling, Unset):
            scheduling = self.scheduling.to_dict()

        search = self.search

        simulations: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.simulations, Unset):
            simulations = []
            for simulations_item_data in self.simulations:
                simulations_item = simulations_item_data.to_dict()
                simulations.append(simulations_item)

        skills: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = []
            for skills_item_data in self.skills:
                skills_item = skills_item_data.to_dict()
                skills.append(skills_item)

        sts = self.sts

        stt = self.stt

        subagent = self.subagent

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        tools: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = self.tools.to_dict()

        tts = self.tts

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "hash": hash_,
                "name": name,
            }
        )
        if base_change is not UNSET:
            field_dict["base_change"] = base_change
        if channels is not UNSET:
            field_dict["channels"] = channels
        if check_changes is not UNSET:
            field_dict["check_changes"] = check_changes
        if connectors is not UNSET:
            field_dict["connectors"] = connectors
        if dispatch is not UNSET:
            field_dict["dispatch"] = dispatch
        if greeting is not UNSET:
            field_dict["greeting"] = greeting
        if guardrail is not UNSET:
            field_dict["guardrail"] = guardrail
        if harness is not UNSET:
            field_dict["harness"] = harness
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if keyterms is not UNSET:
            field_dict["keyterms"] = keyterms
        if knowledge is not UNSET:
            field_dict["knowledge"] = knowledge
        if knowledge_urls is not UNSET:
            field_dict["knowledge_urls"] = knowledge_urls
        if left_out_connectors is not UNSET:
            field_dict["left_out_connectors"] = left_out_connectors
        if llm is not UNSET:
            field_dict["llm"] = llm
        if mcp_servers is not UNSET:
            field_dict["mcp_servers"] = mcp_servers
        if mode is not UNSET:
            field_dict["mode"] = mode
        if plugin_events is not UNSET:
            field_dict["plugin_events"] = plugin_events
        if plugins is not UNSET:
            field_dict["plugins"] = plugins
        if resume_history is not UNSET:
            field_dict["resume_history"] = resume_history
        if role is not UNSET:
            field_dict["role"] = role
        if sandbox is not UNSET:
            field_dict["sandbox"] = sandbox
        if sandbox_options is not UNSET:
            field_dict["sandbox_options"] = sandbox_options
        if scheduling is not UNSET:
            field_dict["scheduling"] = scheduling
        if search is not UNSET:
            field_dict["search"] = search
        if simulations is not UNSET:
            field_dict["simulations"] = simulations
        if skills is not UNSET:
            field_dict["skills"] = skills
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if tags is not UNSET:
            field_dict["tags"] = tags
        if tools is not UNSET:
            field_dict["tools"] = tools
        if tts is not UNSET:
            field_dict["tts"] = tts
        if video is not UNSET:
            field_dict["video"] = video
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_channels import AgentChannels
        from ..models.agent_connector_binding import AgentConnectorBinding
        from ..models.agent_dispatch import AgentDispatch
        from ..models.agent_scheduling import AgentScheduling
        from ..models.agent_tools import AgentTools
        from ..models.greeting import Greeting
        from ..models.knowledge_document import KnowledgeDocument
        from ..models.knowledge_url_declaration import KnowledgeUrlDeclaration
        from ..models.left_out_connector import LeftOutConnector
        from ..models.mcp_server import McpServer
        from ..models.plugin_event import PluginEvent
        from ..models.plugin_with_options import PluginWithOptions
        from ..models.resume_history import ResumeHistory
        from ..models.sandbox_options import SandboxOptions
        from ..models.session_video import SessionVideo
        from ..models.simulation_declaration import SimulationDeclaration
        from ..models.skill_request import SkillRequest
        from ..models.sync_agent_request_tags import SyncAgentRequestTags

        d = dict(src_dict)
        hash_ = d.pop("hash")

        name = d.pop("name")

        base_change = d.pop("base_change", UNSET)

        _channels = d.pop("channels", UNSET)
        channels: AgentChannels | Unset
        if isinstance(_channels, Unset):
            channels = UNSET
        else:
            channels = AgentChannels.from_dict(_channels)

        check_changes = d.pop("check_changes", UNSET)

        _connectors = d.pop("connectors", UNSET)
        connectors: list[AgentConnectorBinding] | Unset = UNSET
        if _connectors is not UNSET:
            connectors = []
            for connectors_item_data in _connectors:
                connectors_item = AgentConnectorBinding.from_dict(connectors_item_data)

                connectors.append(connectors_item)

        _dispatch = d.pop("dispatch", UNSET)
        dispatch: AgentDispatch | Unset
        if isinstance(_dispatch, Unset):
            dispatch = UNSET
        else:
            dispatch = AgentDispatch.from_dict(_dispatch)

        _greeting = d.pop("greeting", UNSET)
        greeting: Greeting | Unset
        if isinstance(_greeting, Unset):
            greeting = UNSET
        else:
            greeting = Greeting.from_dict(_greeting)

        guardrail = d.pop("guardrail", UNSET)

        _harness = d.pop("harness", UNSET)
        harness: Harness | Unset
        if isinstance(_harness, Unset):
            harness = UNSET
        else:
            harness = Harness(_harness)

        instructions = d.pop("instructions", UNSET)

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        _knowledge = d.pop("knowledge", UNSET)
        knowledge: list[KnowledgeDocument] | Unset = UNSET
        if _knowledge is not UNSET:
            knowledge = []
            for knowledge_item_data in _knowledge:
                knowledge_item = KnowledgeDocument.from_dict(knowledge_item_data)

                knowledge.append(knowledge_item)

        _knowledge_urls = d.pop("knowledge_urls", UNSET)
        knowledge_urls: list[KnowledgeUrlDeclaration] | Unset = UNSET
        if _knowledge_urls is not UNSET:
            knowledge_urls = []
            for knowledge_urls_item_data in _knowledge_urls:
                knowledge_urls_item = KnowledgeUrlDeclaration.from_dict(
                    knowledge_urls_item_data
                )

                knowledge_urls.append(knowledge_urls_item)

        _left_out_connectors = d.pop("left_out_connectors", UNSET)
        left_out_connectors: list[LeftOutConnector] | Unset = UNSET
        if _left_out_connectors is not UNSET:
            left_out_connectors = []
            for left_out_connectors_item_data in _left_out_connectors:
                left_out_connectors_item = LeftOutConnector.from_dict(
                    left_out_connectors_item_data
                )

                left_out_connectors.append(left_out_connectors_item)

        llm = d.pop("llm", UNSET)

        _mcp_servers = d.pop("mcp_servers", UNSET)
        mcp_servers: list[McpServer] | Unset = UNSET
        if _mcp_servers is not UNSET:
            mcp_servers = []
            for mcp_servers_item_data in _mcp_servers:
                mcp_servers_item = McpServer.from_dict(mcp_servers_item_data)

                mcp_servers.append(mcp_servers_item)

        _mode = d.pop("mode", UNSET)
        mode: AgentMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = AgentMode(_mode)

        _plugin_events = d.pop("plugin_events", UNSET)
        plugin_events: list[PluginEvent] | Unset = UNSET
        if _plugin_events is not UNSET:
            plugin_events = []
            for plugin_events_item_data in _plugin_events:
                plugin_events_item = PluginEvent.from_dict(plugin_events_item_data)

                plugin_events.append(plugin_events_item)

        _plugins = d.pop("plugins", UNSET)
        plugins: list[PluginWithOptions | str] | Unset = UNSET
        if _plugins is not UNSET:
            plugins = []
            for plugins_item_data in _plugins:

                def _parse_plugins_item(data: object) -> PluginWithOptions | str:
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        componentsschemas_plugin_entry_type_1 = (
                            PluginWithOptions.from_dict(data)
                        )

                        return componentsschemas_plugin_entry_type_1
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    return cast(PluginWithOptions | str, data)

                plugins_item = _parse_plugins_item(plugins_item_data)

                plugins.append(plugins_item)

        _resume_history = d.pop("resume_history", UNSET)
        resume_history: ResumeHistory | Unset
        if isinstance(_resume_history, Unset):
            resume_history = UNSET
        else:
            resume_history = ResumeHistory.from_dict(_resume_history)

        _role = d.pop("role", UNSET)
        role: AgentRole | Unset
        if isinstance(_role, Unset):
            role = UNSET
        else:
            role = AgentRole(_role)

        _sandbox = d.pop("sandbox", UNSET)
        sandbox: Sandbox | Unset
        if isinstance(_sandbox, Unset):
            sandbox = UNSET
        else:
            sandbox = Sandbox(_sandbox)

        _sandbox_options = d.pop("sandbox_options", UNSET)
        sandbox_options: SandboxOptions | Unset
        if isinstance(_sandbox_options, Unset):
            sandbox_options = UNSET
        else:
            sandbox_options = SandboxOptions.from_dict(_sandbox_options)

        _scheduling = d.pop("scheduling", UNSET)
        scheduling: AgentScheduling | Unset
        if isinstance(_scheduling, Unset):
            scheduling = UNSET
        else:
            scheduling = AgentScheduling.from_dict(_scheduling)

        search = d.pop("search", UNSET)

        _simulations = d.pop("simulations", UNSET)
        simulations: list[SimulationDeclaration] | Unset = UNSET
        if _simulations is not UNSET:
            simulations = []
            for simulations_item_data in _simulations:
                simulations_item = SimulationDeclaration.from_dict(
                    simulations_item_data
                )

                simulations.append(simulations_item)

        _skills = d.pop("skills", UNSET)
        skills: list[SkillRequest] | Unset = UNSET
        if _skills is not UNSET:
            skills = []
            for skills_item_data in _skills:
                skills_item = SkillRequest.from_dict(skills_item_data)

                skills.append(skills_item)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: SyncAgentRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SyncAgentRequestTags.from_dict(_tags)

        _tools = d.pop("tools", UNSET)
        tools: AgentTools | Unset
        if isinstance(_tools, Unset):
            tools = UNSET
        else:
            tools = AgentTools.from_dict(_tools)

        tts = d.pop("tts", UNSET)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        voice = d.pop("voice", UNSET)

        sync_agent_request = cls(
            hash_=hash_,
            name=name,
            base_change=base_change,
            channels=channels,
            check_changes=check_changes,
            connectors=connectors,
            dispatch=dispatch,
            greeting=greeting,
            guardrail=guardrail,
            harness=harness,
            instructions=instructions,
            keyterms=keyterms,
            knowledge=knowledge,
            knowledge_urls=knowledge_urls,
            left_out_connectors=left_out_connectors,
            llm=llm,
            mcp_servers=mcp_servers,
            mode=mode,
            plugin_events=plugin_events,
            plugins=plugins,
            resume_history=resume_history,
            role=role,
            sandbox=sandbox,
            sandbox_options=sandbox_options,
            scheduling=scheduling,
            search=search,
            simulations=simulations,
            skills=skills,
            sts=sts,
            stt=stt,
            subagent=subagent,
            tags=tags,
            tools=tools,
            tts=tts,
            video=video,
            voice=voice,
        )

        sync_agent_request.additional_properties = d
        return sync_agent_request

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
