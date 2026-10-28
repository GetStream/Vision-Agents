from __future__ import annotations

import datetime
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
    from ..models.agent_config_tags import AgentConfigTags
    from ..models.agent_connector_binding import AgentConnectorBinding
    from ..models.agent_dispatch import AgentDispatch
    from ..models.agent_review import AgentReview
    from ..models.agent_scheduling import AgentScheduling
    from ..models.agent_tools import AgentTools
    from ..models.greeting import Greeting
    from ..models.left_out_connector import LeftOutConnector
    from ..models.mcp_server import McpServer
    from ..models.plugin_event import PluginEvent
    from ..models.plugin_with_options import PluginWithOptions
    from ..models.resume_history import ResumeHistory
    from ..models.sandbox_options import SandboxOptions
    from ..models.session_video import SessionVideo


T = TypeVar("T", bound="AgentConfig")


@_attrs_define
class AgentConfig:
    """
    Attributes:
        created_at (datetime.datetime):
        id (str):
        mode (AgentMode): Whether the agent is spoken to or written to. A voice agent joins a call, transcribes what it
            hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither speech
            target and a session created from it needs no call to join.
        name (str):
        updated_at (datetime.datetime):
        channels (AgentChannels | Unset): The lines this agent answers on besides its Stream Chat channel. Each names a
            number the app connected with POST /v1/agents/channels, and only one agent may answer on a number. A message
            that arrives is answered in the sender's own conversation, so what they say is kept and shown wherever the rest
            of it is.
        connectors (list[AgentConnectorBinding] | Unset): The bindings exactly as they were written. Absent when there
            are none.
        dispatch (AgentDispatch | Unset): What the agent leaves to the customer's own server, which waits on
            /v1/dispatch. Omitted settings are disabled.
        episode_cards (bool | Unset): Whether each phone call under this agent writes an episode card into the caller's
            omni-channel, and each session on a thread channel or a phone call starts with the person's other cards.
        greeting (Greeting | Unset): What the agent says as it joins, before anyone speaks.
        guardrail (str | Unset):
        harness (Harness | Unset): Which harness the agent's sessions run: what hands work to the subagent, loads
            skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the
            default, the only one there is.
        instructions (str | Unset):
        keyterms (list[str] | Unset):
        knowledge_namespace (str | Unset):
        left_out_connectors (list[LeftOutConnector] | Unset): The connectors agent.yaml names that the last sync left
            out, because the app has no connection by the id or label they name: an agent that runs without them until the
            connection exists and it syncs again. Only a sync writes them. Absent when there are none.
        llm (str | Unset):
        mcp_servers (list[McpServer] | Unset):
        plugin_events (list[PluginEvent] | Unset): Deprecated: use the events of a fixed binding under connectors.
        plugins (list[PluginWithOptions | str] | Unset): Deprecated: use connectors, a binding to a connector.
        resume_history (ResumeHistory | Unset): How a conversation is handed back to the model when a session carries it
            on.
        review (AgentReview | Unset): How a finished call under an agent is reviewed.
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
        skills (list[str] | Unset):
        sts (str | Unset): A speech-to-speech target: one native audio model that hears the caller and speaks back.
            Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade.
        stt (str | Unset):
        subagent (str | Unset):
        sync_hash (str | Unset): Fingerprint of the last directory synced onto this config. Empty if it was never synced
            from a directory.
        tags (AgentConfigTags | Unset):
        tools (AgentTools | Unset): How an agent is offered its tools, and how long what they were given and gave back
            is kept.
        tts (str | Unset):
        video (SessionVideo | Unset):
        visible_tools (list[str] | Unset): Tools whose steps end users see on a persistent conversation's replies.
            Absent when the config never set it, which shows search and web_search; an empty list shows no tool.
        voice (str | Unset): The voice the agent speaks in, as TtsOptions.voice takes it: a provider's own voice id, or
            one of your voices, with the custom: prefix to mean only yours. A native agent (sts) takes the vendor's own
            voice name instead.
    """

    created_at: datetime.datetime
    id: str
    mode: AgentMode
    name: str
    updated_at: datetime.datetime
    channels: AgentChannels | Unset = UNSET
    connectors: list[AgentConnectorBinding] | Unset = UNSET
    dispatch: AgentDispatch | Unset = UNSET
    episode_cards: bool | Unset = UNSET
    greeting: Greeting | Unset = UNSET
    guardrail: str | Unset = UNSET
    harness: Harness | Unset = UNSET
    instructions: str | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    knowledge_namespace: str | Unset = UNSET
    left_out_connectors: list[LeftOutConnector] | Unset = UNSET
    llm: str | Unset = UNSET
    mcp_servers: list[McpServer] | Unset = UNSET
    plugin_events: list[PluginEvent] | Unset = UNSET
    plugins: list[PluginWithOptions | str] | Unset = UNSET
    resume_history: ResumeHistory | Unset = UNSET
    review: AgentReview | Unset = UNSET
    role: AgentRole | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    sandbox_options: SandboxOptions | Unset = UNSET
    scheduling: AgentScheduling | Unset = UNSET
    search: str | Unset = UNSET
    skills: list[str] | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    sync_hash: str | Unset = UNSET
    tags: AgentConfigTags | Unset = UNSET
    tools: AgentTools | Unset = UNSET
    tts: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    visible_tools: list[str] | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.plugin_with_options import PluginWithOptions

        created_at = self.created_at.isoformat()

        id = self.id

        mode = self.mode.value

        name = self.name

        updated_at = self.updated_at.isoformat()

        channels: dict[str, Any] | Unset = UNSET
        if not isinstance(self.channels, Unset):
            channels = self.channels.to_dict()

        connectors: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.connectors, Unset):
            connectors = []
            for connectors_item_data in self.connectors:
                connectors_item = connectors_item_data.to_dict()
                connectors.append(connectors_item)

        dispatch: dict[str, Any] | Unset = UNSET
        if not isinstance(self.dispatch, Unset):
            dispatch = self.dispatch.to_dict()

        episode_cards = self.episode_cards

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

        knowledge_namespace = self.knowledge_namespace

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

        review: dict[str, Any] | Unset = UNSET
        if not isinstance(self.review, Unset):
            review = self.review.to_dict()

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

        skills: list[str] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = self.skills

        sts = self.sts

        stt = self.stt

        subagent = self.subagent

        sync_hash = self.sync_hash

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

        visible_tools: list[str] | Unset = UNSET
        if not isinstance(self.visible_tools, Unset):
            visible_tools = self.visible_tools

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "id": id,
                "mode": mode,
                "name": name,
                "updated_at": updated_at,
            }
        )
        if channels is not UNSET:
            field_dict["channels"] = channels
        if connectors is not UNSET:
            field_dict["connectors"] = connectors
        if dispatch is not UNSET:
            field_dict["dispatch"] = dispatch
        if episode_cards is not UNSET:
            field_dict["episode_cards"] = episode_cards
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
        if knowledge_namespace is not UNSET:
            field_dict["knowledge_namespace"] = knowledge_namespace
        if left_out_connectors is not UNSET:
            field_dict["left_out_connectors"] = left_out_connectors
        if llm is not UNSET:
            field_dict["llm"] = llm
        if mcp_servers is not UNSET:
            field_dict["mcp_servers"] = mcp_servers
        if plugin_events is not UNSET:
            field_dict["plugin_events"] = plugin_events
        if plugins is not UNSET:
            field_dict["plugins"] = plugins
        if resume_history is not UNSET:
            field_dict["resume_history"] = resume_history
        if review is not UNSET:
            field_dict["review"] = review
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
        if skills is not UNSET:
            field_dict["skills"] = skills
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if sync_hash is not UNSET:
            field_dict["sync_hash"] = sync_hash
        if tags is not UNSET:
            field_dict["tags"] = tags
        if tools is not UNSET:
            field_dict["tools"] = tools
        if tts is not UNSET:
            field_dict["tts"] = tts
        if video is not UNSET:
            field_dict["video"] = video
        if visible_tools is not UNSET:
            field_dict["visible_tools"] = visible_tools
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_channels import AgentChannels
        from ..models.agent_config_tags import AgentConfigTags
        from ..models.agent_connector_binding import AgentConnectorBinding
        from ..models.agent_dispatch import AgentDispatch
        from ..models.agent_review import AgentReview
        from ..models.agent_scheduling import AgentScheduling
        from ..models.agent_tools import AgentTools
        from ..models.greeting import Greeting
        from ..models.left_out_connector import LeftOutConnector
        from ..models.mcp_server import McpServer
        from ..models.plugin_event import PluginEvent
        from ..models.plugin_with_options import PluginWithOptions
        from ..models.resume_history import ResumeHistory
        from ..models.sandbox_options import SandboxOptions
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        mode = AgentMode(d.pop("mode"))

        name = d.pop("name")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        _channels = d.pop("channels", UNSET)
        channels: AgentChannels | Unset
        if isinstance(_channels, Unset):
            channels = UNSET
        else:
            channels = AgentChannels.from_dict(_channels)

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

        episode_cards = d.pop("episode_cards", UNSET)

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

        knowledge_namespace = d.pop("knowledge_namespace", UNSET)

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

        _review = d.pop("review", UNSET)
        review: AgentReview | Unset
        if isinstance(_review, Unset):
            review = UNSET
        else:
            review = AgentReview.from_dict(_review)

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

        skills = cast(list[str], d.pop("skills", UNSET))

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        sync_hash = d.pop("sync_hash", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigTags.from_dict(_tags)

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

        visible_tools = cast(list[str], d.pop("visible_tools", UNSET))

        voice = d.pop("voice", UNSET)

        agent_config = cls(
            created_at=created_at,
            id=id,
            mode=mode,
            name=name,
            updated_at=updated_at,
            channels=channels,
            connectors=connectors,
            dispatch=dispatch,
            episode_cards=episode_cards,
            greeting=greeting,
            guardrail=guardrail,
            harness=harness,
            instructions=instructions,
            keyterms=keyterms,
            knowledge_namespace=knowledge_namespace,
            left_out_connectors=left_out_connectors,
            llm=llm,
            mcp_servers=mcp_servers,
            plugin_events=plugin_events,
            plugins=plugins,
            resume_history=resume_history,
            review=review,
            role=role,
            sandbox=sandbox,
            sandbox_options=sandbox_options,
            scheduling=scheduling,
            search=search,
            skills=skills,
            sts=sts,
            stt=stt,
            subagent=subagent,
            sync_hash=sync_hash,
            tags=tags,
            tools=tools,
            tts=tts,
            video=video,
            visible_tools=visible_tools,
            voice=voice,
        )

        agent_config.additional_properties = d
        return agent_config

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
