from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.create_session_request_custom import CreateSessionRequestCustom
    from ..models.create_session_request_tags import CreateSessionRequestTags
    from ..models.history_message import HistoryMessage
    from ..models.model_overwrites import ModelOverwrites
    from ..models.session_memory import SessionMemory
    from ..models.session_phone import SessionPhone
    from ..models.session_tool import SessionTool
    from ..models.session_video import SessionVideo


T = TypeVar("T", bound="CreateSessionRequest")


@_attrs_define
class CreateSessionRequest:
    """
    Attributes:
        agent (str | Unset): The name of an agent config to start from, as an alternative to config_id. It is what a
            caller actually knows the agent as: "docs" rather than an id they never chose. A name matching nothing is
            refused rather than silently starting an unconfigured agent, and naming both this and config_id is refused too,
            since there is no sensible answer when they disagree.
        agent_id (str | Unset): Keys transcripts and statistics. Empty means the call id.
        backchannel (bool | Unset): Murmur while a participant is still talking, the way a person does. Default: False.
        call_id (str | Unset): The call to join. Required unless the session is text.
        call_type (str | Unset):  Default: 'default'.
        config_id (str | Unset): An agent config to start from. Everything else in this request overrides what the
            config says, so a caller can reuse a configuration and still change one thing about this call.
        context_truncated (bool | Unset): Older history was omitted from the model context.
        conversation_id (str | Unset): Stream Chat CID to resume; returned for persistent text sessions.
        custom (CreateSessionRequestCustom | Unset): Anything the caller wants to remember about the session, handed
            back untouched and never read by the router. Sessions can be queried by these, which is what makes them worth
            writing.
        description (str | Unset): A longer note about the conversation, searched alongside the title.
        greeting (str | Unset): Said on joining without going through the model. Empty means the agent waits to be
            spoken to.
        history (list[HistoryMessage] | Unset): The conversation so far, for a backend that keeps its own: a thread in
            its own Slack app, say, that outlives any one session. Send it when a session closed and the thread goes on:
            open a new session with the thread's messages here, oldest first, then send the message to answer to the
            responses endpoint. The model is handed them before the first response, as a resumed conversation's history is.
            They are recorded nowhere, as turns, transcript or Chat messages, so add incognito to keep nothing at all. Up to
            100 messages and 60000 characters of text, the most a session reads back of a conversation the router kept; more
            is refused rather than cut. Not with conversation_id, which reads the history the router kept. Server-side only:
            a device sending it is refused with a 403, because an assistant message puts words in the agent's mouth.
        id (str | Unset): The id to hold the session by, so a caller can know it before the session exists. It must be a
            UUID nobody has used for a session before. Omitted, the router generates a UUIDv7.
        incognito (bool | Unset): Hold the conversation and record nothing about it: no session row, no turns, no
            transcript, and no Stream Chat channel. The session still works exactly as any other while it is running; it
            simply cannot be found afterwards, which is the point. Forking one is refused, because there is nothing to fork
            from. Default: False.
        instructions (str | Unset):
        keyterms (list[str] | Unset): Business-specific words the transcriber would otherwise get wrong. Up to 100
            terms, and providers that cannot be told about vocabulary ignore them.
        languages (list[str] | Unset): Language hints, which narrow the candidates in every modality.
        llm (str | Unset): A provider/model or a capability shortcut. Omit it and the config decides, or llm-fast when
            there is no config. These carry no schema default on purpose: a generated client that filled one in would send
            it, and a caller naming a config would silently lose the model it configured.
        max_tokens (int | Unset):
        memory (SessionMemory | Unset): Who the session's memories are about. Without a user id nothing is recalled or
            stored, which is the case for a call with nobody identified on it.
        min_confidence (float | Unset): How sure the transcriber must be before the agent answers rather than checks
            what was meant.
        model_overwrites (ModelOverwrites | Unset): What to change about the models for one session, over whatever its
            agent config decided.
            It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides
            the config, and a caller reading a session back wants to see what they changed in one place rather than diffed
            against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because
            a caller able to rewrite those could make a session impersonate a different agent.
        navigating (bool | Unset): The agent placed this call, so let recordings finish and answer their menus. Default:
            False.
        phone (SessionPhone | Unset): The number the session acts from, which is what turns transferring on.
        project_id (str | Unset): What the conversation belongs to. Also recorded as the "project" cost tag, so spend
            breaks down by project without the caller labelling it twice. A tag spelled out in tags wins.
        search (str | Unset): Omit it and the config decides, or search-fast when there is no config.
        sts (str | Unset): A speech-to-speech target. Naming one makes this a native session: the model hears and speaks
            for itself, so no transcriber, conversation model or voice is opened. Omit it and the config decides.
        stt (str | Unset): Omit it and the config decides, or en-low-latency when there is no config.
        tags (CreateSessionRequestTags | Unset): Cost labels, carried onto every request the session makes.
        text (bool | Unset): Hold the conversation in writing rather than on a call. Nothing is transcribed and nothing
            is spoken, so no call is joined and neither speech target is used. Everything between hearing and answering is
            unchanged: a text session has the same skills, knowledge and tools a call would have had, and its replies arrive
            as response_delta and responded events on the session's socket. Default: False.
        title (str | Unset): What to call the conversation, for a list a person reads, until the router names a
            persistent one for what was said. Never shown to the model: what a conversation is called is a label on it
            rather than part of it.
        tool_timeout_ms (int | Unset): How long the model waits for a tool result. Zero is the default.
        tools (list[SessionTool] | Unset):
        tts (str | Unset): Omit it and the config decides, or en-low-latency when there is no config.
        user_id (str | Unset): Who the agent joins the call as. Default: 'vision-agent'.
        user_name (str | Unset):  Default: 'Vision Agent'.
        video (SessionVideo | Unset):
        voice (str | Unset): Provider-specific voice id.
    """

    agent: str | Unset = UNSET
    agent_id: str | Unset = UNSET
    backchannel: bool | Unset = False
    call_id: str | Unset = UNSET
    call_type: str | Unset = "default"
    config_id: str | Unset = UNSET
    context_truncated: bool | Unset = UNSET
    conversation_id: str | Unset = UNSET
    custom: CreateSessionRequestCustom | Unset = UNSET
    description: str | Unset = UNSET
    greeting: str | Unset = UNSET
    history: list[HistoryMessage] | Unset = UNSET
    id: str | Unset = UNSET
    incognito: bool | Unset = False
    instructions: str | Unset = UNSET
    keyterms: list[str] | Unset = UNSET
    languages: list[str] | Unset = UNSET
    llm: str | Unset = UNSET
    max_tokens: int | Unset = UNSET
    memory: SessionMemory | Unset = UNSET
    min_confidence: float | Unset = UNSET
    model_overwrites: ModelOverwrites | Unset = UNSET
    navigating: bool | Unset = False
    phone: SessionPhone | Unset = UNSET
    project_id: str | Unset = UNSET
    search: str | Unset = UNSET
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    tags: CreateSessionRequestTags | Unset = UNSET
    text: bool | Unset = False
    title: str | Unset = UNSET
    tool_timeout_ms: int | Unset = UNSET
    tools: list[SessionTool] | Unset = UNSET
    tts: str | Unset = UNSET
    user_id: str | Unset = "vision-agent"
    user_name: str | Unset = "Vision Agent"
    video: SessionVideo | Unset = UNSET
    voice: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent = self.agent

        agent_id = self.agent_id

        backchannel = self.backchannel

        call_id = self.call_id

        call_type = self.call_type

        config_id = self.config_id

        context_truncated = self.context_truncated

        conversation_id = self.conversation_id

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        description = self.description

        greeting = self.greeting

        history: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.history, Unset):
            history = []
            for history_item_data in self.history:
                history_item = history_item_data.to_dict()
                history.append(history_item)

        id = self.id

        incognito = self.incognito

        instructions = self.instructions

        keyterms: list[str] | Unset = UNSET
        if not isinstance(self.keyterms, Unset):
            keyterms = self.keyterms

        languages: list[str] | Unset = UNSET
        if not isinstance(self.languages, Unset):
            languages = self.languages

        llm = self.llm

        max_tokens = self.max_tokens

        memory: dict[str, Any] | Unset = UNSET
        if not isinstance(self.memory, Unset):
            memory = self.memory.to_dict()

        min_confidence = self.min_confidence

        model_overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.model_overwrites, Unset):
            model_overwrites = self.model_overwrites.to_dict()

        navigating = self.navigating

        phone: dict[str, Any] | Unset = UNSET
        if not isinstance(self.phone, Unset):
            phone = self.phone.to_dict()

        project_id = self.project_id

        search = self.search

        sts = self.sts

        stt = self.stt

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        text = self.text

        title = self.title

        tool_timeout_ms = self.tool_timeout_ms

        tools: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = []
            for tools_item_data in self.tools:
                tools_item = tools_item_data.to_dict()
                tools.append(tools_item)

        tts = self.tts

        user_id = self.user_id

        user_name = self.user_name

        video: dict[str, Any] | Unset = UNSET
        if not isinstance(self.video, Unset):
            video = self.video.to_dict()

        voice = self.voice

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if agent is not UNSET:
            field_dict["agent"] = agent
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if backchannel is not UNSET:
            field_dict["backchannel"] = backchannel
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if call_type is not UNSET:
            field_dict["call_type"] = call_type
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if context_truncated is not UNSET:
            field_dict["context_truncated"] = context_truncated
        if conversation_id is not UNSET:
            field_dict["conversation_id"] = conversation_id
        if custom is not UNSET:
            field_dict["custom"] = custom
        if description is not UNSET:
            field_dict["description"] = description
        if greeting is not UNSET:
            field_dict["greeting"] = greeting
        if history is not UNSET:
            field_dict["history"] = history
        if id is not UNSET:
            field_dict["id"] = id
        if incognito is not UNSET:
            field_dict["incognito"] = incognito
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if keyterms is not UNSET:
            field_dict["keyterms"] = keyterms
        if languages is not UNSET:
            field_dict["languages"] = languages
        if llm is not UNSET:
            field_dict["llm"] = llm
        if max_tokens is not UNSET:
            field_dict["max_tokens"] = max_tokens
        if memory is not UNSET:
            field_dict["memory"] = memory
        if min_confidence is not UNSET:
            field_dict["min_confidence"] = min_confidence
        if model_overwrites is not UNSET:
            field_dict["model_overwrites"] = model_overwrites
        if navigating is not UNSET:
            field_dict["navigating"] = navigating
        if phone is not UNSET:
            field_dict["phone"] = phone
        if project_id is not UNSET:
            field_dict["project_id"] = project_id
        if search is not UNSET:
            field_dict["search"] = search
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tags is not UNSET:
            field_dict["tags"] = tags
        if text is not UNSET:
            field_dict["text"] = text
        if title is not UNSET:
            field_dict["title"] = title
        if tool_timeout_ms is not UNSET:
            field_dict["tool_timeout_ms"] = tool_timeout_ms
        if tools is not UNSET:
            field_dict["tools"] = tools
        if tts is not UNSET:
            field_dict["tts"] = tts
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if user_name is not UNSET:
            field_dict["user_name"] = user_name
        if video is not UNSET:
            field_dict["video"] = video
        if voice is not UNSET:
            field_dict["voice"] = voice

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.create_session_request_custom import (
            CreateSessionRequestCustom,
        )
        from ..models.create_session_request_tags import (
            CreateSessionRequestTags,
        )
        from ..models.history_message import HistoryMessage
        from ..models.model_overwrites import ModelOverwrites
        from ..models.session_memory import SessionMemory
        from ..models.session_phone import SessionPhone
        from ..models.session_tool import SessionTool
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        agent = d.pop("agent", UNSET)

        agent_id = d.pop("agent_id", UNSET)

        backchannel = d.pop("backchannel", UNSET)

        call_id = d.pop("call_id", UNSET)

        call_type = d.pop("call_type", UNSET)

        config_id = d.pop("config_id", UNSET)

        context_truncated = d.pop("context_truncated", UNSET)

        conversation_id = d.pop("conversation_id", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: CreateSessionRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = CreateSessionRequestCustom.from_dict(_custom)

        description = d.pop("description", UNSET)

        greeting = d.pop("greeting", UNSET)

        _history = d.pop("history", UNSET)
        history: list[HistoryMessage] | Unset = UNSET
        if _history is not UNSET:
            history = []
            for history_item_data in _history:
                history_item = HistoryMessage.from_dict(history_item_data)

                history.append(history_item)

        id = d.pop("id", UNSET)

        incognito = d.pop("incognito", UNSET)

        instructions = d.pop("instructions", UNSET)

        keyterms = cast(list[str], d.pop("keyterms", UNSET))

        languages = cast(list[str], d.pop("languages", UNSET))

        llm = d.pop("llm", UNSET)

        max_tokens = d.pop("max_tokens", UNSET)

        _memory = d.pop("memory", UNSET)
        memory: SessionMemory | Unset
        if isinstance(_memory, Unset):
            memory = UNSET
        else:
            memory = SessionMemory.from_dict(_memory)

        min_confidence = d.pop("min_confidence", UNSET)

        _model_overwrites = d.pop("model_overwrites", UNSET)
        model_overwrites: ModelOverwrites | Unset
        if isinstance(_model_overwrites, Unset):
            model_overwrites = UNSET
        else:
            model_overwrites = ModelOverwrites.from_dict(_model_overwrites)

        navigating = d.pop("navigating", UNSET)

        _phone = d.pop("phone", UNSET)
        phone: SessionPhone | Unset
        if isinstance(_phone, Unset):
            phone = UNSET
        else:
            phone = SessionPhone.from_dict(_phone)

        project_id = d.pop("project_id", UNSET)

        search = d.pop("search", UNSET)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: CreateSessionRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CreateSessionRequestTags.from_dict(_tags)

        text = d.pop("text", UNSET)

        title = d.pop("title", UNSET)

        tool_timeout_ms = d.pop("tool_timeout_ms", UNSET)

        _tools = d.pop("tools", UNSET)
        tools: list[SessionTool] | Unset = UNSET
        if _tools is not UNSET:
            tools = []
            for tools_item_data in _tools:
                tools_item = SessionTool.from_dict(tools_item_data)

                tools.append(tools_item)

        tts = d.pop("tts", UNSET)

        user_id = d.pop("user_id", UNSET)

        user_name = d.pop("user_name", UNSET)

        _video = d.pop("video", UNSET)
        video: SessionVideo | Unset
        if isinstance(_video, Unset):
            video = UNSET
        else:
            video = SessionVideo.from_dict(_video)

        voice = d.pop("voice", UNSET)

        create_session_request = cls(
            agent=agent,
            agent_id=agent_id,
            backchannel=backchannel,
            call_id=call_id,
            call_type=call_type,
            config_id=config_id,
            context_truncated=context_truncated,
            conversation_id=conversation_id,
            custom=custom,
            description=description,
            greeting=greeting,
            history=history,
            id=id,
            incognito=incognito,
            instructions=instructions,
            keyterms=keyterms,
            languages=languages,
            llm=llm,
            max_tokens=max_tokens,
            memory=memory,
            min_confidence=min_confidence,
            model_overwrites=model_overwrites,
            navigating=navigating,
            phone=phone,
            project_id=project_id,
            search=search,
            sts=sts,
            stt=stt,
            tags=tags,
            text=text,
            title=title,
            tool_timeout_ms=tool_timeout_ms,
            tools=tools,
            tts=tts,
            user_id=user_id,
            user_name=user_name,
            video=video,
            voice=voice,
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
