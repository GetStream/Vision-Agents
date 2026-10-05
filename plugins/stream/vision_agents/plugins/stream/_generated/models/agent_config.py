from __future__ import annotations

import datetime
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
    from ..models.agent_config_tags import AgentConfigTags
    from ..models.agent_connector_binding import AgentConnectorBinding
    from ..models.agent_dispatch import AgentDispatch
    from ..models.mcp_server import McpServer
    from ..models.plugin_event import PluginEvent
    from ..models.plugin_with_options import PluginWithOptions
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
        agent_plugins (list[PluginWithOptions | str] | Unset):
        channels (AgentChannels | Unset): The lines this agent answers on besides its Stream Chat channel. Each names a
            number the app connected with POST /v1/agents/channels, and only one agent may answer on a number. A message
            that arrives is answered in the sender's own conversation, so what they say is kept and shown wherever the rest
            of it is.
        connectors (list[AgentConnectorBinding] | Unset): The bindings exactly as they were written. Absent when there
            are none.
        dispatch (AgentDispatch | Unset): What the agent leaves to the customer's own server, which waits on
            /v1/dispatch. Omitted settings are disabled.
        greeting (str | Unset):
        guardrail (str | Unset):
        harness (Harness | Unset): Which harness the agent's sessions run: what hands work to the subagent, loads
            skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the
            default, the only one there is.
        instructions (str | Unset):
        keyterms (list[str] | Unset):
        knowledge_namespace (str | Unset):
        llm (str | Unset):
        mcp_servers (list[McpServer] | Unset):
        plugin_events (list[PluginEvent] | Unset):
        sandbox (Sandbox | Unset): Where the subagent may run code it writes. Only the subagent is offered it: running
            code takes seconds, and the model holding the conversation has none to spare. Omit it and the subagent works
            everything out in its head.
        sandbox_options (SandboxOptions | Unset): How the sandbox is built and how long code may run in it. Only
            meaningful with a sandbox. Omit it for the provider's own Python sandbox and a 30 second run.
        search (str | Unset):
        skills (list[str] | Unset):
        speed (float | Unset):
        sts (str | Unset): A speech-to-speech target: one native audio model that hears the caller and speaks back.
            Naming one makes the agent native, and stt, tts and llm are then not used. Empty means the cascade.
        stt (str | Unset):
        sync_hash (str | Unset): Fingerprint of the last directory synced onto this config. Empty if it was never synced
            from a directory.
        tags (AgentConfigTags | Unset):
        thinking_llm (str | Unset):
        tts (str | Unset):
        user_plugins (list[PluginWithOptions | str] | Unset):
        video (SessionVideo | Unset):
        visible_tools (list[str] | Unset):
        voice (str | Unset):
    """

    created_at: datetime.datetime
    id: str
    mode: AgentMode
    name: str
    updated_at: datetime.datetime
    agent_plugins: list[PluginWithOptions | str] | Unset = UNSET
    channels: AgentChannels | Unset = UNSET
    connectors: list[AgentConnectorBinding] | Unset = UNSET
    dispatch: AgentDispatch | Unset = UNSET
    greeting: str | Unset = UNSET
    guardrail: str | Unset = UNSET
    harness: Harness | Unset = UNSET
    instructions: str | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    knowledge_namespace: str | Unset = UNSET
    llm: str | Unset = UNSET
    mcp_servers: list[McpServer] | Unset = UNSET
    plugin_events: list[PluginEvent] | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    sandbox_options: SandboxOptions | Unset = UNSET
    search: str | Unset = UNSET
    skills: list[str] | Unset = UNSET
    speed: float | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    sync_hash: str | Unset = UNSET
    tags: AgentConfigTags | Unset = UNSET
    thinking_llm: str | Unset = UNSET
    tts: str | Unset = UNSET
    user_plugins: list[PluginWithOptions | str] | Unset = UNSET
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

        agent_plugins: list[dict[str, Any] | str] | Unset = UNSET
        if not isinstance(self.agent_plugins, Unset):
            agent_plugins = []
            for agent_plugins_item_data in self.agent_plugins:
                agent_plugins_item: dict[str, Any] | str
                if isinstance(agent_plugins_item_data, PluginWithOptions):
                    agent_plugins_item = agent_plugins_item_data.to_dict()
                else:
                    agent_plugins_item = agent_plugins_item_data
                agent_plugins.append(agent_plugins_item)

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

        plugin_events: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.plugin_events, Unset):
            plugin_events = []
            for plugin_events_item_data in self.plugin_events:
                plugin_events_item = plugin_events_item_data.to_dict()
                plugin_events.append(plugin_events_item)

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

        sync_hash = self.sync_hash

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        thinking_llm = self.thinking_llm

        tts = self.tts

        user_plugins: list[dict[str, Any] | str] | Unset = UNSET
        if not isinstance(self.user_plugins, Unset):
            user_plugins = []
            for user_plugins_item_data in self.user_plugins:
                user_plugins_item: dict[str, Any] | str
                if isinstance(user_plugins_item_data, PluginWithOptions):
                    user_plugins_item = user_plugins_item_data.to_dict()
                else:
                    user_plugins_item = user_plugins_item_data
                user_plugins.append(user_plugins_item)

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
        if agent_plugins is not UNSET:
            field_dict["agent_plugins"] = agent_plugins
        if channels is not UNSET:
            field_dict["channels"] = channels
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
        if knowledge_namespace is not UNSET:
            field_dict["knowledge_namespace"] = knowledge_namespace
        if llm is not UNSET:
            field_dict["llm"] = llm
        if mcp_servers is not UNSET:
            field_dict["mcp_servers"] = mcp_servers
        if plugin_events is not UNSET:
            field_dict["plugin_events"] = plugin_events
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
        if sync_hash is not UNSET:
            field_dict["sync_hash"] = sync_hash
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
        from ..models.agent_channels import AgentChannels
        from ..models.agent_config_tags import AgentConfigTags
        from ..models.agent_connector_binding import (
            AgentConnectorBinding,
        )
        from ..models.agent_dispatch import AgentDispatch
        from ..models.mcp_server import McpServer
        from ..models.plugin_event import PluginEvent
        from ..models.plugin_with_options import PluginWithOptions
        from ..models.sandbox_options import SandboxOptions
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        mode = AgentMode(d.pop("mode"))

        name = d.pop("name")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        _agent_plugins = d.pop("agent_plugins", UNSET)
        agent_plugins: list[PluginWithOptions | str] | Unset = UNSET
        if _agent_plugins is not UNSET:
            agent_plugins = []
            for agent_plugins_item_data in _agent_plugins:

                def _parse_agent_plugins_item(data: object) -> PluginWithOptions | str:
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

                agent_plugins_item = _parse_agent_plugins_item(agent_plugins_item_data)

                agent_plugins.append(agent_plugins_item)

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

        _plugin_events = d.pop("plugin_events", UNSET)
        plugin_events: list[PluginEvent] | Unset = UNSET
        if _plugin_events is not UNSET:
            plugin_events = []
            for plugin_events_item_data in _plugin_events:
                plugin_events_item = PluginEvent.from_dict(plugin_events_item_data)

                plugin_events.append(plugin_events_item)

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

        sync_hash = d.pop("sync_hash", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigTags.from_dict(_tags)

        thinking_llm = d.pop("thinking_llm", UNSET)

        tts = d.pop("tts", UNSET)

        _user_plugins = d.pop("user_plugins", UNSET)
        user_plugins: list[PluginWithOptions | str] | Unset = UNSET
        if _user_plugins is not UNSET:
            user_plugins = []
            for user_plugins_item_data in _user_plugins:

                def _parse_user_plugins_item(data: object) -> PluginWithOptions | str:
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

                user_plugins_item = _parse_user_plugins_item(user_plugins_item_data)

                user_plugins.append(user_plugins_item)

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
            agent_plugins=agent_plugins,
            channels=channels,
            connectors=connectors,
            dispatch=dispatch,
            greeting=greeting,
            guardrail=guardrail,
            harness=harness,
            instructions=instructions,
            keyterms=keyterms,
            knowledge_namespace=knowledge_namespace,
            llm=llm,
            mcp_servers=mcp_servers,
            plugin_events=plugin_events,
            sandbox=sandbox,
            sandbox_options=sandbox_options,
            search=search,
            skills=skills,
            speed=speed,
            sts=sts,
            stt=stt,
            sync_hash=sync_hash,
            tags=tags,
            thinking_llm=thinking_llm,
            tts=tts,
            user_plugins=user_plugins,
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
