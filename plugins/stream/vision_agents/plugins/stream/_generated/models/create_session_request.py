from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.create_session_request_sandbox import CreateSessionRequestSandbox
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.create_session_request_tags import CreateSessionRequestTags
    from ..models.session_memory import SessionMemory
    from ..models.session_phone import SessionPhone
    from ..models.session_skill import SessionSkill
    from ..models.session_tool import SessionTool


T = TypeVar("T", bound="CreateSessionRequest")


@_attrs_define
class CreateSessionRequest:
    """
    Attributes:
        call_id (str | Unset): The call to join. Required unless the session is text.
        text (bool | Unset): Hold the conversation in writing rather than on a call. Nothing is transcribed and nothing
            is spoken, so no call is joined and neither speech target is used. Everything between hearing and answering is
            unchanged: a text session has the same skills, knowledge and tools a call would have had, and its replies arrive
            as response_delta and responded events on the session's socket.
             Default: False.
        config_id (str | Unset): An agent config to start from. Everything else in this request overrides what the
            config says, so a caller can reuse a configuration and still change one thing about this call.
        call_type (str | Unset):  Default: 'default'.
        user_id (str | Unset): Who the agent joins the call as. Default: 'vision-agent'.
        user_name (str | Unset):  Default: 'Vision Agent'.
        agent_id (str | Unset): Keys transcripts and statistics. Empty means the call id.
        instructions (str | Unset):
        greeting (str | Unset): Said on joining without going through the model. Empty means the agent waits to be
            spoken to.
        navigating (bool | Unset): The agent placed this call, so let recordings finish and answer their menus.
             Default: False.
        llm (str | Unset): A provider/model or a capability shortcut. Omit it and the config decides, or llm-fast when
            there is no config. These carry no schema default on purpose: a generated client that filled one in would send
            it, and a caller naming a config would silently lose the model it configured.
        stt (str | Unset): Omit it and the config decides, or en-low-latency when there is no config.
        tts (str | Unset): Omit it and the config decides, or en-low-latency when there is no config.
        subagent (str | Unset): The model that does the thinking. Empty means the voice model answers everything itself,
            and skills mean nothing.
        search (str | Unset): Omit it and the config decides, or search-fast when there is no config.
        voice (str | Unset): Provider-specific voice id.
        languages (list[str] | Unset): Language hints, which narrow the candidates in every modality.
        keyterms (list[str] | Unset): Business-specific words the transcriber would otherwise get wrong. Up to 100
            terms, and providers that cannot be told about vocabulary ignore them.
        max_tokens (int | Unset):
        tasks (int | Unset): How much delegated work may run at once.
        sandbox (CreateSessionRequestSandbox | Unset): Where the subagent may run code it writes. Only the subagent is
            offered it: running code takes seconds, and the model holding the conversation has none to spare. Omit it and
            the subagent works everything out in its head.
        backchannel (bool | Unset): Murmur while a participant is still talking, the way a person does. Default: False.
        min_confidence (float | Unset): How sure the transcriber must be before the agent answers rather than checks
            what was meant.
        skills (list[SessionSkill] | Unset): Omit for the built-in set of think, recall and explain.
        skill_names (list[str] | Unset): Skills to look up rather than spell out: the customer's own, or one of the
            built-in think, recall and explain. Ignored when skills are given in full, and a name nothing defines is refused
            rather than dropped.
        tools (list[SessionTool] | Unset):
        tool_timeout_ms (int | Unset): How long the model waits for a tool result. Zero is the default.
        tags (CreateSessionRequestTags | Unset): Cost labels, carried onto every request the session makes.
        memory (SessionMemory | Unset): Who the session's memories are about. Without a user id nothing is recalled or
            stored, which is the case for a call with nobody identified on it.
        phone (SessionPhone | Unset): The number the session acts from, which is what turns transferring on.
    """

    call_id: str | Unset = UNSET
    text: bool | Unset = False
    config_id: str | Unset = UNSET
    call_type: str | Unset = "default"
    user_id: str | Unset = "vision-agent"
    user_name: str | Unset = "Vision Agent"
    agent_id: str | Unset = UNSET
    instructions: str | Unset = UNSET
    greeting: str | Unset = UNSET
    navigating: bool | Unset = False
    llm: str | Unset = UNSET
    stt: str | Unset = UNSET
    tts: str | Unset = UNSET
    subagent: str | Unset = UNSET
    search: str | Unset = UNSET
    voice: str | Unset = UNSET
    languages: list[str] | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    max_tokens: int | Unset = UNSET
    tasks: int | Unset = UNSET
    sandbox: CreateSessionRequestSandbox | Unset = UNSET
    backchannel: bool | Unset = False
    min_confidence: float | Unset = UNSET
    skills: list[SessionSkill] | Unset = UNSET
    skill_names: list[str] | Unset = UNSET
    tools: list[SessionTool] | Unset = UNSET
    tool_timeout_ms: int | Unset = UNSET
    tags: CreateSessionRequestTags | Unset = UNSET
    memory: SessionMemory | Unset = UNSET
    phone: SessionPhone | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        call_id = self.call_id

        text = self.text

        config_id = self.config_id

        call_type = self.call_type

        user_id = self.user_id

        user_name = self.user_name

        agent_id = self.agent_id

        instructions = self.instructions

        greeting = self.greeting

        navigating = self.navigating

        llm = self.llm

        stt = self.stt

        tts = self.tts

        subagent = self.subagent

        search = self.search

        voice = self.voice

        languages: list[str] | Unset = UNSET
        if not isinstance(self.languages, Unset):
            languages = self.languages

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

        max_tokens = self.max_tokens

        tasks = self.tasks

        sandbox: str | Unset = UNSET
        if not isinstance(self.sandbox, Unset):
            sandbox = self.sandbox.value

        backchannel = self.backchannel

        min_confidence = self.min_confidence

        skills: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = []
            for skills_item_data in self.skills:
                skills_item = skills_item_data.to_dict()
                skills.append(skills_item)

        skill_names: list[str] | Unset = UNSET
        if not isinstance(self.skill_names, Unset):
            skill_names = self.skill_names

        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        tool_timeout_ms = self.tool_timeout_ms

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        memory: dict[str, Any] | Unset = UNSET
        if not isinstance(self.memory, Unset):
            memory = self.memory.to_dict()

        phone: dict[str, Any] | Unset = UNSET
        if not isinstance(self.phone, Unset):
            phone = self.phone.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if text is not UNSET:
            field_dict["text"] = text
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if call_type is not UNSET:
            field_dict["call_type"] = call_type
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if user_name is not UNSET:
            field_dict["user_name"] = user_name
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if greeting is not UNSET:
            field_dict["greeting"] = greeting
        if navigating is not UNSET:
            field_dict["navigating"] = navigating
        if llm is not UNSET:
            field_dict["llm"] = llm
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tts is not UNSET:
            field_dict["tts"] = tts
        if subagent is not UNSET:
            field_dict["subagent"] = subagent
        if search is not UNSET:
            field_dict["search"] = search
        if voice is not UNSET:
            field_dict["voice"] = voice
        if languages is not UNSET:
            field_dict["languages"] = languages
        if keyterms is not UNSET:
            field_dict["keyterms"] = keyterms
        if max_tokens is not UNSET:
            field_dict["max_tokens"] = max_tokens
        if tasks is not UNSET:
            field_dict["tasks"] = tasks
        if sandbox is not UNSET:
            field_dict["sandbox"] = sandbox
        if backchannel is not UNSET:
            field_dict["backchannel"] = backchannel
        if min_confidence is not UNSET:
            field_dict["min_confidence"] = min_confidence
        if skills is not UNSET:
            field_dict["skills"] = skills
        if skill_names is not UNSET:
            field_dict["skill_names"] = skill_names
        if tools is not UNSET:
            field_dict["tools"] = tools
        if tool_timeout_ms is not UNSET:
            field_dict["tool_timeout_ms"] = tool_timeout_ms
        if tags is not UNSET:
            field_dict["tags"] = tags
        if memory is not UNSET:
            field_dict["memory"] = memory
        if phone is not UNSET:
            field_dict["phone"] = phone

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.create_session_request_tags import (
            CreateSessionRequestTags,
        )
        from ..models.session_memory import SessionMemory
        from ..models.session_phone import SessionPhone
        from ..models.session_skill import SessionSkill
        from ..models.session_tool import SessionTool

        d = dict(src_dict)
        call_id = d.pop("call_id", UNSET)

        text = d.pop("text", UNSET)

        config_id = d.pop("config_id", UNSET)

        call_type = d.pop("call_type", UNSET)

        user_id = d.pop("user_id", UNSET)

        user_name = d.pop("user_name", UNSET)

        agent_id = d.pop("agent_id", UNSET)

        instructions = d.pop("instructions", UNSET)

        greeting = d.pop("greeting", UNSET)

        navigating = d.pop("navigating", UNSET)

        llm = d.pop("llm", UNSET)

        stt = d.pop("stt", UNSET)

        tts = d.pop("tts", UNSET)

        subagent = d.pop("subagent", UNSET)

        search = d.pop("search", UNSET)

        voice = d.pop("voice", UNSET)

        languages = cast(list[str], d.pop("languages", UNSET))

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        max_tokens = d.pop("max_tokens", UNSET)

        tasks = d.pop("tasks", UNSET)

        _sandbox = d.pop("sandbox", UNSET)
        sandbox: CreateSessionRequestSandbox | Unset
        if isinstance(_sandbox, Unset):
            sandbox = UNSET
        else:
            sandbox = CreateSessionRequestSandbox(_sandbox)

        backchannel = d.pop("backchannel", UNSET)

        min_confidence = d.pop("min_confidence", UNSET)

        _skills = d.pop("skills", UNSET)
        skills: list[SessionSkill] | Unset = UNSET
        if _skills is not UNSET:
            skills = []
            for skills_item_data in _skills:
                skills_item = SessionSkill.from_dict(skills_item_data)

                skills.append(skills_item)

        skill_names = cast(list[str], d.pop("skill_names", UNSET))

        _tools = d.pop("tools", UNSET)
        tools: list[SessionTool] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = SessionTool.from_dict(tools_item_data)

                tools.append(tools_item)

        tool_timeout_ms = d.pop("tool_timeout_ms", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: CreateSessionRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CreateSessionRequestTags.from_dict(_tags)

        _memory = d.pop("memory", UNSET)
        memory: SessionMemory | Unset
        if isinstance(_memory, Unset):
            memory = UNSET
        else:
            memory = SessionMemory.from_dict(_memory)

        _phone = d.pop("phone", UNSET)
        phone: SessionPhone | Unset
        if isinstance(_phone, Unset):
            phone = UNSET
        else:
            phone = SessionPhone.from_dict(_phone)

        create_session_request = cls(
            call_id=call_id,
            text=text,
            config_id=config_id,
            call_type=call_type,
            user_id=user_id,
            user_name=user_name,
            agent_id=agent_id,
            instructions=instructions,
            greeting=greeting,
            navigating=navigating,
            llm=llm,
            stt=stt,
            tts=tts,
            subagent=subagent,
            search=search,
            voice=voice,
            languages=languages,
            keyterms=keyterms,
            max_tokens=max_tokens,
            tasks=tasks,
            sandbox=sandbox,
            backchannel=backchannel,
            min_confidence=min_confidence,
            skills=skills,
            skill_names=skill_names,
            tools=tools,
            tool_timeout_ms=tool_timeout_ms,
            tags=tags,
            memory=memory,
            phone=phone,
        )

        create_session_request.additional_properties = d
        return create_session_request

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
