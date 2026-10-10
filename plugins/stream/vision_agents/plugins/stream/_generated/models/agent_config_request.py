from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.agent_mode import AgentMode
from ..models.harness import Harness
from ..models.sandbox import Sandbox
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_channels import AgentChannels
    from ..models.agent_config_request_tags import AgentConfigRequestTags
    from ..models.agent_connector_binding import AgentConnectorBinding
    from ..models.agent_dispatch import AgentDispatch
    from ..models.agent_tools import AgentTools
    from ..models.greeting import Greeting
    from ..models.mcp_server import McpServer
    from ..models.plugin_event import PluginEvent
    from ..models.plugin_with_options import PluginWithOptions
    from ..models.sandbox_options import SandboxOptions
    from ..models.session_video import SessionVideo


T = TypeVar("T", bound="AgentConfigRequest")


@_attrs_define
class AgentConfigRequest:
    """
    Attributes:
        name (str): What the config is called, which is unique among the customer's own.
        channels (AgentChannels | Unset): The lines this agent answers on besides its Stream Chat channel. Each names a
            number the app connected with POST /v1/agents/channels, and only one agent may answer on a number. A message
            that arrives is answered in the sender's own conversation, so what they say is kept and shown wherever the rest
            of it is.
        connectors (list[AgentConnectorBinding] | Unset): The connectors whose tools this agent may call, each under an
            alias unique within the config and different from every plugin and MCP server it names. Omitted or null on an
            update, the bindings stored stay as they are, so a client that does not know this field cannot clear it by
            saving; an empty list removes them all. A binding to a connector the app cannot see, or a fixed binding to a
            connection that is not the app's own or is to another connector, is refused.
        dispatch (AgentDispatch | Unset): What the agent leaves to the customer's own server, which waits on
            /v1/dispatch. Omitted settings are disabled.
        episode_cards (bool | Unset): Whether each phone call under this agent writes an episode card into the caller's
            omni-channel: an agent channel for each caller number and agent, keyed by the caller's E.164 number. On, a
            session on a thread channel or a phone call under this agent also starts with the person's other episode cards:
            a summary, or the last lines of the episode's channel while there is none. Off by default, and then a session
            runs as it always did. Left out on an update, the stored setting stays.
        greeting (Greeting | Unset): What the agent says as it joins, before anyone speaks.
        guardrail (str | Unset): A guardrail.md: frontmatter saying how a turn is screened - decision_model, webhook or
            llm - then the policy in prose. A turn the policy refuses is answered with the refusal and never reaches the
            model. Empty means every turn is answered.
        harness (Harness | Unset): Which harness the agent's sessions run: what hands work to the subagent, loads
            skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the
            default, the only one there is.
        instructions (str | Unset):
        keyterms (list[str] | Unset): Business-specific words the transcriber would otherwise get wrong, such as product
            or company names. Up to 100 terms, and providers that cannot be told about vocabulary ignore them.
        knowledge_namespace (str | Unset): What the agent may look things up in. Empty means it knows only what it was
            told.
        llm (str | Unset): The model holding the conversation.
        mcp_servers (list[McpServer] | Unset): MCP servers outside the plugin catalog, opened by their URL with no
            login. Their tools are offered as <name>__<tool>.
        mode (AgentMode | Unset): Whether the agent is spoken to or written to. A voice agent joins a call, transcribes
            what it hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither
            speech target and a session created from it needs no call to join.
        plugin_events (list[PluginEvent] | Unset): Deprecated: use the events of a fixed binding under connectors. MCP
            events the agent subscribes to on the plugins it names, with every login it holds to each. Each event that
            arrives opens a text conversation of its own, as whoever's login it came through.
        plugins (list[PluginWithOptions | str] | Unset): Deprecated: use connectors, a binding to a connector. Hosted
            MCP servers this agent may reach, named from the built-in catalog: an id alone, or an object naming it with how
            it is reached, such as linear's read-only endpoint and the scopes its login asks for. The app connects each
            once, from the dashboard, unless its entry sets user: then each end user connects it with their own account, and
            the agent asks for the login in the conversation, as a plugin_authorization attachment, the first time it needs
            one.
        sandbox (Sandbox | Unset): Where the subagent may run code it writes. Only the subagent is offered it: running
            code takes seconds, and the model holding the conversation has none to spare. Omit it and the subagent works
            everything out in its head.
        sandbox_options (SandboxOptions | Unset): How the sandbox is built and how long code may run in it. Only
            meaningful with a sandbox. Omit it for the provider's own Python sandbox and a 30 second run.
        search (str | Unset): What the agent finds out today's answers with, as a provider/model or a capability
            shortcut. Empty leaves the default, and a deployment that routes no search offers the tool to nobody either way.
        skills (list[str] | Unset): Skill names, either the customer's own or one of the built-in think, recall and
            explain. Omit for the built-in set.
        sts (str | Unset): A speech-to-speech target: one native audio model that hears the caller and speaks back.
            Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade.
        stt (str | Unset): A provider/model or a capability shortcut. Empty leaves the default, and a text agent ignores
            it.
        subagent (str | Unset): The slower model a voice agent hands its skills to, while the voice model keeps talking.
            Only a voice agent names one: a text agent runs everything, skills included, on its llm. Empty leaves the
            default subagent.
        tags (AgentConfigRequestTags | Unset): Cost labels, carried onto every request a session using it makes. A
            config tagged draft_of, naming the config it copies, is a test copy: it answers no message on a channel
            connection and subscribes to no event, which stay with the live config, and it may bind a channel connection
            another config binds.
        tools (AgentTools | Unset): How an agent is offered its plugin, MCP server and connector tools.
        tts (str | Unset):
        video (SessionVideo | Unset):
        visible_tools (list[str] | Unset): Tools whose steps end users see on a persistent conversation's replies, as
            tool names or path.Match patterns such as athena_*. Only a step's name, status and timing are shown, never its
            arguments or result. A shown tool whose result is exactly
            {"status":"answered","citations":[{"id","title","url","citation"}]} also adds those citations to the reply's
            sources. Empty shows search and web_search.
        voice (str | Unset): Provider-specific voice id.
    """

    name: str
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
    llm: str | Unset = UNSET
    mcp_servers: list[McpServer] | Unset = UNSET
    mode: AgentMode | Unset = UNSET
    plugin_events: list[PluginEvent] | Unset = UNSET
    plugins: list[PluginWithOptions | str] | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    sandbox_options: SandboxOptions | Unset = UNSET
    search: str | Unset = UNSET
    skills: list[str] | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    tags: AgentConfigRequestTags | Unset = UNSET
    tools: AgentTools | Unset = UNSET
    tts: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    visible_tools: list[str] | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.plugin_with_options import PluginWithOptions

        name = self.name

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

        sandbox: str | Unset = UNSET
        if not isinstance(self.sandbox, Unset):
            sandbox = self.sandbox.value

        sandbox_options: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sandbox_options, Unset):
            sandbox_options = self.sandbox_options.to_dict()

        search = self.search

        skills: list[str] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = self.skills

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

        visible_tools: list[str] | Unset = UNSET
        if not isinstance(self.visible_tools, Unset):
            visible_tools = self.visible_tools

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
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
        if sandbox is not UNSET:
            field_dict["sandbox"] = sandbox
        if sandbox_options is not UNSET:
            field_dict["sandbox_options"] = sandbox_options
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
        from ..models.agent_config_request_tags import AgentConfigRequestTags
        from ..models.agent_connector_binding import AgentConnectorBinding
        from ..models.agent_dispatch import AgentDispatch
        from ..models.agent_tools import AgentTools
        from ..models.greeting import Greeting
        from ..models.mcp_server import McpServer
        from ..models.plugin_event import PluginEvent
        from ..models.plugin_with_options import PluginWithOptions
        from ..models.sandbox_options import SandboxOptions
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        name = d.pop("name")

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

        search = d.pop("search", UNSET)

        skills = cast(list[str], d.pop("skills", UNSET))

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        subagent = d.pop("subagent", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigRequestTags.from_dict(_tags)

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

        agent_config_request = cls(
            name=name,
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
            llm=llm,
            mcp_servers=mcp_servers,
            mode=mode,
            plugin_events=plugin_events,
            plugins=plugins,
            sandbox=sandbox,
            sandbox_options=sandbox_options,
            search=search,
            skills=skills,
            sts=sts,
            stt=stt,
            subagent=subagent,
            tags=tags,
            tools=tools,
            tts=tts,
            video=video,
            visible_tools=visible_tools,
            voice=voice,
        )

        agent_config_request.additional_properties = d
        return agent_config_request

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
