from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.agent_mode import AgentMode
from ..models.harness import Harness
from ..models.sandbox import Sandbox
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_config_patch_tags import AgentConfigPatchTags
    from ..models.agent_dispatch import AgentDispatch
    from ..models.sandbox_options import SandboxOptions
    from ..models.session_video import SessionVideo


T = TypeVar("T", bound="AgentConfigPatch")


@_attrs_define
class AgentConfigPatch:
    """What changes about an agent config. A field left out keeps what is stored, and an unknown one is refused rather than
    ignored.

        Attributes:
            dispatch (AgentDispatch | Unset): What the agent leaves to the customer's own server, which waits on
                /v1/dispatch. Omitted settings are disabled.
            greeting (str | Unset):
            guardrail (str | Unset): A guardrail.md: frontmatter saying how a turn is screened, then the policy in prose. An
                empty string removes the guardrail.
            harness (Harness | Unset): Which harness the agent's sessions run: what hands work to the subagent, loads
                skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the
                default, the only one there is.
            instructions (str | Unset):
            keyterms (list[str] | Unset):
            knowledge_namespace (str | Unset):
            llm (str | Unset):
            mode (AgentMode | Unset): Whether the agent is spoken to or written to. A voice agent joins a call, transcribes
                what it hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither
                speech target and a session created from it needs no call to join.
            name (str | Unset): What the config is called, which is unique among the customer's own.
            plugins (list[str] | Unset):
            sandbox (Sandbox | Unset): Where the subagent may run code it writes. Only the subagent is offered it: running
                code takes seconds, and the model holding the conversation has none to spare. Omit it and the subagent works
                everything out in its head.
            sandbox_options (SandboxOptions | Unset): How the sandbox is built and how long code may run in it. Only
                meaningful with a sandbox. Omit it for the provider's own Python sandbox and a 30 second run.
            search (str | Unset):
            skills (list[str] | Unset):
            speed (float | Unset): The voice's rate of delivery, 1 being its own. Zero leaves it there.
            sts (str | Unset):
            stt (str | Unset):
            subagent (str | Unset):
            tags (AgentConfigPatchTags | Unset):
            tts (str | Unset):
            user_plugins (list[str] | Unset):
            video (SessionVideo | Unset):
            visible_tools (list[str] | Unset): Tools whose steps end users see on a persistent conversation's replies, as
                tool names or path.Match patterns such as athena_*. Only a step's name, status and timing are shown, never its
                arguments or result. A shown tool whose result is exactly {"status":"answered","citations":[...]} also adds
                those citations to the reply's sources. An empty list shows search and web_search.
            voice (str | Unset):
    """

    dispatch: AgentDispatch | Unset = UNSET
    greeting: str | Unset = UNSET
    guardrail: str | Unset = UNSET
    harness: Harness | Unset = UNSET
    instructions: str | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    knowledge_namespace: str | Unset = UNSET
    llm: str | Unset = UNSET
    mode: AgentMode | Unset = UNSET
    name: str | Unset = UNSET
    plugins: list[str] | Unset = UNSET
    sandbox: Sandbox | Unset = UNSET
    sandbox_options: SandboxOptions | Unset = UNSET
    search: str | Unset = UNSET
    skills: list[str] | Unset = UNSET
    speed: float | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    subagent: str | Unset = UNSET
    tags: AgentConfigPatchTags | Unset = UNSET
    tts: str | Unset = UNSET
    user_plugins: list[str] | Unset = UNSET
    video: SessionVideo | Unset = UNSET
    visible_tools: list[str] | Unset = UNSET
    voice: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
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

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        name = self.name

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

        subagent = self.subagent

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

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

        field_dict.update({})
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
        if mode is not UNSET:
            field_dict["mode"] = mode
        if name is not UNSET:
            field_dict["name"] = name
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
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if tags is not UNSET:
            field_dict["tags"] = tags
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
        from ..models.agent_config_patch_tags import (
            AgentConfigPatchTags,
        )
        from ..models.agent_dispatch import AgentDispatch
        from ..models.sandbox_options import SandboxOptions
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
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

        _mode = d.pop("mode", UNSET)
        mode: AgentMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = AgentMode(_mode)

        name = d.pop("name", UNSET)

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

        subagent = d.pop("subagent", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: AgentConfigPatchTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = AgentConfigPatchTags.from_dict(_tags)

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

        agent_config_patch = cls(
            dispatch=dispatch,
            greeting=greeting,
            guardrail=guardrail,
            harness=harness,
            instructions=instructions,
            keyterms=keyterms,
            knowledge_namespace=knowledge_namespace,
            llm=llm,
            mode=mode,
            name=name,
            plugins=plugins,
            sandbox=sandbox,
            sandbox_options=sandbox_options,
            search=search,
            skills=skills,
            speed=speed,
            sts=sts,
            stt=stt,
            subagent=subagent,
            tags=tags,
            tts=tts,
            user_plugins=user_plugins,
            video=video,
            visible_tools=visible_tools,
            voice=voice,
        )

        return agent_config_patch
