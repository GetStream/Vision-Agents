from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.agent_mode import AgentMode
from ..models.agent_role import AgentRole
from ..models.episode_cards_mode import EpisodeCardsMode
from ..models.harness import Harness
from ..models.sandbox import Sandbox
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_calls import AgentCalls
    from ..models.agent_channels import AgentChannels
    from ..models.agent_chat_channel import AgentChatChannel
    from ..models.agent_config_patch_mcp_servers_item import (
        AgentConfigPatchMcpServersItem,
    )
    from ..models.agent_config_patch_plugin_events_item import (
        AgentConfigPatchPluginEventsItem,
    )
    from ..models.agent_config_patch_plugins_item_type_1 import (
        AgentConfigPatchPluginsItemType1,
    )
    from ..models.agent_config_patch_tags import AgentConfigPatchTags
    from ..models.agent_connector_binding import AgentConnectorBinding
    from ..models.agent_dispatch import AgentDispatch
    from ..models.agent_review import AgentReview
    from ..models.agent_scheduling import AgentScheduling
    from ..models.agent_tools import AgentTools
    from ..models.child_agent import ChildAgent
    from ..models.floor_control import FloorControl
    from ..models.greeting import Greeting
    from ..models.resume_history import ResumeHistory
    from ..models.sandbox_options import SandboxOptions
    from ..models.session_video import SessionVideo


T = TypeVar("T", bound="AgentConfigPatch")


@_attrs_define
class AgentConfigPatch:
    """What changes about an agent config. A field left out keeps what is stored, and an unknown one is refused rather than
    ignored.

        Attributes:
            calls (AgentCalls | Unset): The tools an agent on a phone call is offered: end the call, press keys, transfer.
            channels (AgentChannels | Unset): The lines this agent answers on besides its Stream Chat channel. Each names a
                number the app connected with POST /v1/agents/channels, and only one agent may answer on a number. A message
                that arrives is answered in the sender's own conversation, so what they say is kept and shown wherever the rest
                of it is.
            connectors (list[AgentConnectorBinding] | Unset): The connectors whose tools the agent may call, each under an
                alias unique within the config. Sent, they replace the bindings stored, and an empty list removes them all. Null
                is the same as leaving them out.
            dispatch (AgentDispatch | Unset): What the agent leaves to the customer's own server, which waits on
                /v1/dispatch. Omitted settings are disabled.
            episode_cards (EpisodeCardsMode | Unset): What the agent does with episode cards, the same for every channel:
                its phone calls, the app chat, and the SMS, WhatsApp, iMessage and Slack threads of its channel connections. off
                writes none and reads none. record writes each conversation's card and its summary into the person's omni-
                channel, and no session reads them. on writes them as record does, and a session starts with the person's other
                cards. Turning record into on reads what record wrote.
            floor_control (FloorControl | Unset): The model that controls an agent's floor, and how long a caller may wait
                once they stop. Left out or null, the stored choice stays; an empty object is no choice again, which the
                router's default rules: the flow controller, or stream/chorus-asr at the server's default point where the router
                defaults every agent to it; {model: flow-controller} keeps the flow controller whatever the router's default;
                anything else replaces the stored choice whole.
            greeting (Greeting | Unset): What the agent says as it joins, before anyone speaks.
            guardrail (str | Unset): A guardrail.md: frontmatter saying how a turn is screened, then the policy in prose. An
                empty string removes the guardrail.
            harness (Harness | Unset): Which harness the agent's sessions run: what hands work to the subagent, loads
                skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the
                default, the only one there is.
            instructions (str | Unset):
            keyterms (list[str] | Unset):
            knowledge_namespace (str | Unset):
            llm (str | Unset):
            mcp_servers (list[AgentConfigPatchMcpServersItem] | Unset): No longer read: a server named by URL is now a
                custom connector, so create one for its URL and name it under connectors. A non-empty list is refused (400); an
                empty one names nothing.
            mode (AgentMode | Unset): Whether the agent is spoken to or written to. A voice agent joins a call, transcribes
                what it hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither
                speech target and a session created from it needs no call to join.
            name (str | Unset): What the config is called, which is unique among the customer's own.
            plugin_events (list[AgentConfigPatchPluginEventsItem] | Unset): No longer read: plugins were replaced by
                connectors, so name the provider under connectors. A non-empty list is refused (400); an empty one names
                nothing.
            plugins (list[AgentConfigPatchPluginsItemType1 | str] | Unset): No longer read: plugins were replaced by
                connectors, so name the provider under connectors. A non-empty list is refused (400); an empty one names
                nothing.
            region (str | Unset): Where the agent's work is done: us or eu for now. A change naming a model with no instance
                there, a knowledge base kept in another region, or a sandbox where none runs there is refused (400), naming
                each. Example: eu.
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
            sts (str | Unset):
            stt (str | Unset):
            subagent (str | Unset): Only a voice agent names one. Switching an agent to text drops it.
            tags (AgentConfigPatchTags | Unset):
            tools (AgentTools | Unset): How an agent is offered its tools, and how long what they were given and gave back
                is kept.
            tts (str | Unset):
            video (SessionVideo | Unset):
            visible_tools (list[str] | Unset): Tools whose steps end users see on a persistent conversation's replies, as
                tool names or path.Match patterns such as athena_*. Only a step's name, status and timing are shown, never its
                arguments or result. A shown tool whose result is exactly {"status":"answered","citations":[...]} also adds
                those citations to the reply's sources. An empty list shows search and web_search.
            voice (str | Unset): The voice the agent speaks in, as TtsOptions.voice takes it: a provider's own voice id, or
                one of your voices, with the custom: prefix to mean only yours. A native agent (sts) takes the vendor's own
                voice name instead.
            llm_fallbacks (list[str] | Unset): Replaces the list whole. An empty list removes it.
            chat_channels (list[AgentChatChannel] | Unset): Replaces the list whole. An empty list removes it.
            agents (list[ChildAgent] | Unset): Replaces the list whole. An empty list removes it.
            image (str | Unset): An empty string removes it, and the agent stops being offered generate_image.
    """

    calls: AgentCalls | Unset = UNSET
    channels: AgentChannels | Unset = UNSET
    connectors: list[AgentConnectorBinding] | Unset = UNSET
    dispatch: AgentDispatch | Unset = UNSET
    episode_cards: EpisodeCardsMode | Unset = UNSET
    floor_control: FloorControl | Unset = UNSET
    greeting: Greeting | Unset = UNSET
    guardrail: str | Unset = UNSET
    harness: Harness | Unset = UNSET
    instructions: str | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    knowledge_namespace: str | Unset = UNSET
    llm: str | Unset = UNSET
    mcp_servers: list[AgentConfigPatchMcpServersItem] | Unset = UNSET
    mode: AgentMode | Unset = UNSET
    name: str | Unset = UNSET
    plugin_events: list[AgentConfigPatchPluginEventsItem] | Unset = UNSET
    plugins: list[AgentConfigPatchPluginsItemType1 | str] | Unset = UNSET
    region: str | Unset = UNSET
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
    tags: AgentConfigPatchTags | Unset = UNSET
    tools: AgentTools | Unset = UNSET
    tts: str | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    visible_tools: list[str] | Unset = UNSET
    voice: str | Unset = UNSET
    llm_fallbacks: list[str] | Unset = UNSET
    chat_channels: list[AgentChatChannel] | Unset = UNSET
    agents: list[ChildAgent] | Unset = UNSET
    image: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.agent_config_patch_plugins_item_type_1 import (
            AgentConfigPatchPluginsItemType1,
        )

        calls: dict[str, Any] | Unset = UNSET
        if not isinstance(self.calls, Unset):
            calls = self.calls.to_dict()

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

        episode_cards: str | Unset = UNSET
        if not isinstance(self.episode_cards, Unset):
            episode_cards = self.episode_cards.value

        floor_control: dict[str, Any] | Unset = UNSET
        if not isinstance(self.floor_control, Unset):
            floor_control = self.floor_control.to_dict()

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

        name = self.name

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
                if isinstance(plugins_item_data, AgentConfigPatchPluginsItemType1):
                    plugins_item = plugins_item_data.to_dict()
                else:
                    plugins_item = plugins_item_data
                plugins.append(plugins_item)

        region = self.region

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

        llm_fallbacks: list[str] | Unset = UNSET
        if not isinstance(self.llm_fallbacks, Unset):
            llm_fallbacks = self.llm_fallbacks

        chat_channels: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.chat_channels, Unset):
            chat_channels = []
            for chat_channels_item_data in self.chat_channels:
                chat_channels_item = chat_channels_item_data.to_dict()
                chat_channels.append(chat_channels_item)

        agents: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.agents, Unset):
            agents = []
            for agents_item_data in self.agents:
                agents_item = agents_item_data.to_dict()
                agents.append(agents_item)

        image = self.image

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if calls is not UNSET:
            field_dict["calls"] = calls
        if channels is not UNSET:
            field_dict["channels"] = channels
        if connectors is not UNSET:
            field_dict["connectors"] = connectors
        if dispatch is not UNSET:
            field_dict["dispatch"] = dispatch
        if episode_cards is not UNSET:
            field_dict["episode_cards"] = episode_cards
        if floor_control is not UNSET:
            field_dict["floor_control"] = floor_control
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
        if name is not UNSET:
            field_dict["name"] = name
        if plugin_events is not UNSET:
            field_dict["plugin_events"] = plugin_events
        if plugins is not UNSET:
            field_dict["plugins"] = plugins
        if region is not UNSET:
            field_dict["region"] = region
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
        if llm_fallbacks is not UNSET:
            field_dict["llm_fallbacks"] = llm_fallbacks
        if chat_channels is not UNSET:
            field_dict["chat_channels"] = chat_channels
        if agents is not UNSET:
            field_dict["agents"] = agents
        if image is not UNSET:
            field_dict["image"] = image

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_calls import AgentCalls
        from ..models.agent_channels import AgentChannels
        from ..models.agent_chat_channel import AgentChatChannel
        from ..models.agent_config_patch_mcp_servers_item import (
            AgentConfigPatchMcpServersItem,
        )
        from ..models.agent_config_patch_plugin_events_item import (
            AgentConfigPatchPluginEventsItem,
        )
        from ..models.agent_config_patch_plugins_item_type_1 import (
            AgentConfigPatchPluginsItemType1,
        )
        from ..models.agent_config_patch_tags import AgentConfigPatchTags
        from ..models.agent_connector_binding import AgentConnectorBinding
        from ..models.agent_dispatch import AgentDispatch
        from ..models.agent_review import AgentReview
        from ..models.agent_scheduling import AgentScheduling
        from ..models.agent_tools import AgentTools
        from ..models.child_agent import ChildAgent
        from ..models.floor_control import FloorControl
        from ..models.greeting import Greeting
        from ..models.resume_history import ResumeHistory
        from ..models.sandbox_options import SandboxOptions
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        _calls = d.pop("calls", UNSET)
        calls: AgentCalls | Unset
        if isinstance(_calls, Unset):
            calls = UNSET
        else:
            calls = AgentCalls.from_dict(_calls)

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

        _episode_cards = d.pop("episode_cards", UNSET)
        episode_cards: EpisodeCardsMode | Unset
        if isinstance(_episode_cards, Unset):
            episode_cards = UNSET
        else:
            episode_cards = EpisodeCardsMode._decode(_episode_cards)

        _floor_control = d.pop("floor_control", UNSET)
        floor_control: FloorControl | Unset
        if isinstance(_floor_control, Unset):
            floor_control = UNSET
        else:
            floor_control = FloorControl.from_dict(_floor_control)

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
            harness = Harness._decode(_harness)

        instructions = d.pop("instructions", UNSET)

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        knowledge_namespace = d.pop("knowledge_namespace", UNSET)

        llm = d.pop("llm", UNSET)

        _mcp_servers = d.pop("mcp_servers", UNSET)
        mcp_servers: list[AgentConfigPatchMcpServersItem] | Unset = UNSET
        if _mcp_servers is not UNSET:
            mcp_servers = []
            for mcp_servers_item_data in _mcp_servers:
                mcp_servers_item = AgentConfigPatchMcpServersItem.from_dict(
                    mcp_servers_item_data
                )

                mcp_servers.append(mcp_servers_item)

        _mode = d.pop("mode", UNSET)
        mode: AgentMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = AgentMode._decode(_mode)

        name = d.pop("name", UNSET)

        _plugin_events = d.pop("plugin_events", UNSET)
        plugin_events: list[AgentConfigPatchPluginEventsItem] | Unset = UNSET
        if _plugin_events is not UNSET:
            plugin_events = []
            for plugin_events_item_data in _plugin_events:
                plugin_events_item = AgentConfigPatchPluginEventsItem.from_dict(
                    plugin_events_item_data
                )

                plugin_events.append(plugin_events_item)

        _plugins = d.pop("plugins", UNSET)
        plugins: list[AgentConfigPatchPluginsItemType1 | str] | Unset = UNSET
        if _plugins is not UNSET:
            plugins = []
            for plugins_item_data in _plugins:

                def _parse_plugins_item(
                    data: object,
                ) -> AgentConfigPatchPluginsItemType1 | str:
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        plugins_item_type_1 = (
                            AgentConfigPatchPluginsItemType1.from_dict(data)
                        )

                        return plugins_item_type_1
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    return cast(AgentConfigPatchPluginsItemType1 | str, data)

                plugins_item = _parse_plugins_item(plugins_item_data)

                plugins.append(plugins_item)

        region = d.pop("region", UNSET)

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
            role = AgentRole._decode(_role)

        _sandbox = d.pop("sandbox", UNSET)
        sandbox: Sandbox | Unset
        if isinstance(_sandbox, Unset):
            sandbox = UNSET
        else:
            sandbox = Sandbox._decode(_sandbox)

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

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigPatchTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigPatchTags.from_dict(_tags)

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

        llm_fallbacks = cast(list[str], d.pop("llm_fallbacks", UNSET))

        _chat_channels = d.pop("chat_channels", UNSET)
        chat_channels: list[AgentChatChannel] | Unset = UNSET
        if _chat_channels is not UNSET:
            chat_channels = []
            for chat_channels_item_data in _chat_channels:
                chat_channels_item = AgentChatChannel.from_dict(chat_channels_item_data)

                chat_channels.append(chat_channels_item)

        _agents = d.pop("agents", UNSET)
        agents: list[ChildAgent] | Unset = UNSET
        if _agents is not UNSET:
            agents = []
            for agents_item_data in _agents:
                agents_item = ChildAgent.from_dict(agents_item_data)

                agents.append(agents_item)

        image = d.pop("image", UNSET)

        agent_config_patch = cls(
            calls=calls,
            channels=channels,
            connectors=connectors,
            dispatch=dispatch,
            episode_cards=episode_cards,
            floor_control=floor_control,
            greeting=greeting,
            guardrail=guardrail,
            harness=harness,
            instructions=instructions,
            keyterms=keyterms,
            knowledge_namespace=knowledge_namespace,
            llm=llm,
            mcp_servers=mcp_servers,
            mode=mode,
            name=name,
            plugin_events=plugin_events,
            plugins=plugins,
            region=region,
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
            tags=tags,
            tools=tools,
            tts=tts,
            video=video,
            visible_tools=visible_tools,
            voice=voice,
            llm_fallbacks=llm_fallbacks,
            chat_channels=chat_channels,
            agents=agents,
            image=image,
        )

        return agent_config_patch
