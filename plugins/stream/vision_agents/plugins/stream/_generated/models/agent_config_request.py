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
    from ..models.agent_config_request_tags import AgentConfigRequestTags
    from ..models.agent_dispatch import AgentDispatch
    from ..models.mcp_server import McpServer
    from ..models.plugin_event import PluginEvent
    from ..models.sandbox_options import SandboxOptions
    from ..models.session_video import SessionVideo


T = TypeVar("T", bound="AgentConfigRequest")


@_attrs_define
class AgentConfigRequest:
    """
    Attributes:
        name (str): What the config is called, which is unique among the customer's own.
        dispatch (AgentDispatch | Unset): What the agent leaves to the customer's own server, which waits on
            /v1/dispatch. Omitted settings are disabled.
        greeting (str | Unset):
        guardrail (str | Unset): A guardrail.md: frontmatter saying how a turn is screened - lcm, webhook or llm - then
            the policy in prose. A turn the policy refuses is answered with the refusal and never reaches the model. Empty
            means every turn is answered.
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
        plugin_events (list[PluginEvent] | Unset): MCP events the agent subscribes to on the plugins it names, with
            every login it holds to each. Each event that arrives opens a text conversation of its own, as whoever's login
            it came through.
        plugins (list[str] | Unset): Hosted MCP servers this agent may reach, named from the built-in catalog.
        sandbox (Sandbox | Unset): Where the subagent may run code it writes. Only the subagent is offered it: running
            code takes seconds, and the model holding the conversation has none to spare. Omit it and the subagent works
            everything out in its head.
        sandbox_options (SandboxOptions | Unset): How the sandbox is built and how long code may run in it. Only
            meaningful with a sandbox. Omit it for the provider's own Python sandbox and a 30 second run.
        search (str | Unset): What the agent finds out today's answers with, as a provider/model or a capability
            shortcut. Empty leaves the default, and a deployment that routes no search offers the tool to nobody either way.
        skills (list[str] | Unset): Skill names, either the customer's own or one of the built-in think, recall and
            explain. Omit for the built-in set.
        speed (float | Unset): Rate of delivery, 1 being the voice's own. Zero or absent leaves it there. A config that
            names one is only routed to voices that can be sped up, and one outside that voice's own range is refused.
            Example: 0.9.
        sts (str | Unset): A speech-to-speech target: one native audio model that hears the caller and speaks back.
            Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade.
        stt (str | Unset): A provider/model or a capability shortcut. Empty leaves the default, and a text agent ignores
            it.
        tags (AgentConfigRequestTags | Unset): Cost labels, carried onto every request a session using it makes.
        thinking_llm (str | Unset): The slower model a voice agent hands its skills to, while the voice model keeps
            talking. Only a voice agent names one: a text agent runs everything, skills included, on its llm. Empty leaves
            the default thinking model.
        tts (str | Unset):
        user_plugins (list[str] | Unset): Hosted MCP servers each end user connects with their own account, named from
            the built-in catalog. The agent asks for the login in the conversation, as a plugin_authorization attachment,
            the first time it needs one.
        video (SessionVideo | Unset):
        visible_tools (list[str] | Unset): Tools whose steps end users see on a persistent conversation's replies, as
            tool names or path.Match patterns such as athena_*. Only a step's name, status and timing are shown, never its
            arguments or result. A shown tool whose result is exactly
            {"status":"answered","citations":[{"id","title","url","citation"}]} also adds those citations to the reply's
            sources. Empty shows search and web_search.
        voice (str | Unset): Provider-specific voice id.
    """

    name: str
    dispatch: AgentDispatch | Unset = UNSET
    greeting: str | Unset = UNSET
    guardrail: str | Unset = UNSET
    harness: Harness | Unset = UNSET
    instructions: str | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    knowledge_namespace: str | Unset = UNSET
    llm: str | Unset = UNSET
    mcp_servers: list[McpServer] | Unset = UNSET
    mode: AgentMode | Unset = UNSET
    plugin_events: list[PluginEvent] | Unset = UNSET
    plugins: list[str] | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    sandbox_options: SandboxOptions | Unset = UNSET
    search: str | Unset = UNSET
    skills: list[str] | Unset = UNSET
    speed: float | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    tags: AgentConfigRequestTags | Unset = UNSET
    thinking_llm: str | Unset = UNSET
    tts: str | Unset = UNSET
    user_plugins: list[str] | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    visible_tools: list[str] | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        dispatch: dict[str, Any] | Unset = UNSET
        if not isinstance(self.dispatch, Unset):
            dispatch = self.dispatch.to_dict()

        greeting = self.greeting

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

        plugins: list[str] | Unset = UNSET
        if not isinstance(self.plugins, Unset):
            plugins = self.plugins

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

        speed = self.speed

        sts = self.sts

        stt = self.stt

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        thinking_llm = self.thinking_llm

        tts = self.tts

        user_plugins: list[str] | Unset = UNSET
        if not isinstance(self.user_plugins, Unset):
            user_plugins = self.user_plugins

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
        if speed is not UNSET:
            field_dict["speed"] = speed
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tags is not UNSET:
            field_dict["tags"] = tags
        if thinking_llm is not UNSET:
            field_dict["thinking_llm"] = thinking_llm
        if tts is not UNSET:
            field_dict["tts"] = tts
        if user_plugins is not UNSET:
            field_dict["user_plugins"] = user_plugins
        if video is not UNSET:
            field_dict["video"] = video
        if visible_tools is not UNSET:
            field_dict["visible_tools"] = visible_tools
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_config_request_tags import (
            AgentConfigRequestTags,
        )
        from ..models.agent_dispatch import AgentDispatch
        from ..models.mcp_server import McpServer
        from ..models.plugin_event import PluginEvent
        from ..models.sandbox_options import SandboxOptions
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        name = d.pop("name")

        _dispatch = d.pop("dispatch", UNSET)
        dispatch: AgentDispatch | Unset
        if isinstance(_dispatch, Unset):
            dispatch = UNSET
        else:
            dispatch = AgentDispatch.from_dict(_dispatch)

        greeting = d.pop("greeting", UNSET)

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

        plugins = cast(list[str], d.pop("plugins", UNSET))

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

        speed = d.pop("speed", UNSET)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigRequestTags.from_dict(_tags)

        thinking_llm = d.pop("thinking_llm", UNSET)

        tts = d.pop("tts", UNSET)

        user_plugins = cast(list[str], d.pop("user_plugins", UNSET))

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
            dispatch=dispatch,
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
            speed=speed,
            sts=sts,
            stt=stt,
            tags=tags,
            thinking_llm=thinking_llm,
            tts=tts,
            user_plugins=user_plugins,
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
