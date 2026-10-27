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
    from ..models.greeting import Greeting
    from ..models.history_message import HistoryMessage
    from ..models.model_overwrites import ModelOverwrites
    from ..models.resume_history import ResumeHistory
    from ..models.session_call import SessionCall
    from ..models.session_connector_binding import SessionConnectorBinding
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
        call (SessionCall | Unset): An existing Stream call for the agent to join, by its type and id.
        config_id (str | Unset): An agent config to start from. Everything else in this request overrides what the
            config says, so a caller can reuse a configuration and still change one thing about this call.
        connector_bindings (list[SessionConnectorBinding] | Unset): The connection to use for each of the agent config's
            connector bindings chosen per session (connection.type session), by its alias. Each must be the verified
            caller's own connection to the binding's connector: an end user's, or the one a backend names with X-Stream-
            User-Id, never an anonymous caller's or a guest's. A binding with a fixed connection cannot be given one here,
            and an alias the config does not declare is refused. A session binding given none here uses the caller's own
            connection to its connector when exactly one of theirs is connected. Otherwise, with none connected or more than
            one, a required binding fails the session; an optional one is left out and reported with a connector_unavailable
            event (no_selection). A fork chooses the same connections again, against the config as it is then and the caller
            asking for the fork.
        context_truncated (bool | Unset): Older history was omitted from the model context.
        custom (CreateSessionRequestCustom | Unset): Anything the caller wants to remember about the session, handed
            back untouched and never read by the router. Sessions can be queried by these, which is what makes them worth
            writing.
        description (str | Unset): A longer note about the conversation, searched alongside the title.
        greeting (Greeting | Unset): What the agent says as it joins, before anyone speaks.
        history (list[HistoryMessage] | Unset): The conversation so far, for a backend that keeps its own: a thread in
            its own Slack app, say, that outlives any one session. Send it when a session closed and the thread goes on:
            open a new session with the thread's messages here, oldest first, then send the message to answer to the
            responses endpoint. The model is handed them before the first response, as a resumed conversation's history is.
            They are recorded nowhere, as turns, transcript or Chat messages, so add incognito to keep nothing at all. Up to
            100 messages and 60000 characters of text, the most a session reads back of a conversation the router kept; more
            is refused rather than cut. Server-side only: a device sending it is refused with a 403, because an assistant
            message puts words in the agent's mouth.
        id (str | Unset): The id to hold the session by, so a caller can know it before the session exists. It may be
            any string of up to 64 letters, digits, - and _, such as your own record's id, that nobody has used for a
            session before. A UUID is held lowercase. Omitted, the router generates a UUIDv7.
        incognito (bool | Unset): Hold the conversation and record nothing about it: no session row, no turns, no
            transcript, and no Stream Chat channel. The session still works exactly as any other while it is running; it
            simply cannot be found afterwards, which is the point. Forking one is refused, because there is nothing to fork
            from. Default: False.
        instructions (str | Unset): What the agent is told to be for this session alone, in place of its config's
            instructions: a role given at call time, such as the brief of a customer somebody practises on, without a stored
            config per brief. Empty or left out, the config's instructions stand. There is no limit but the request body's,
            as a config's instructions have none. They are held only by the running session: a session carried on in writing
            after it ended in another process, a fork of a session no longer running and a fork naming a config run on the
            config's instructions again. Server-side only: a device sending them is refused with a 403, because what the
            agent is told decides who it is, and a device could make it impersonate a different agent.
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
            a caller able to rewrite those could make a session impersonate a different agent. A backend sets one session's
            instructions with the instructions of CreateSessionRequest instead, which a device may not send.
        navigating (bool | Unset): The agent placed this call, so let recordings finish and answer their menus. Default:
            False.
        phone (SessionPhone | Unset): The number the session acts from, which is what turns transferring on.
        project_id (str | Unset): What the conversation belongs to. Also recorded as the "project" cost tag, so spend
            breaks down by project without the caller labelling it twice. A tag spelled out in tags wins.
        resume_history (ResumeHistory | Unset): How a conversation is handed back to the model when a session carries it
            on.
        return_url (str | Unset): The app's own page a login or step-up the chat asks for goes back to, as
            createAuthorization's return_url: the launch page takes the handoff only from a page of its origin, and the
            browser comes back to it with connection_id and status added to its query. The same rules as
            createAuthorization's return_url apply, and a URL they refuse is answered with a 400 invalid_request naming the
            rule (an error frame on the session socket). Absent or empty, those consents go back to the dashboard. It is
            kept with the session, so a chat carried on after it ended and a fork go back to it too. An end user's device
            may send it: a consent the chat asks for is only ever for the caller's own connection, shown to the caller
            alone, and bound to the first browser that hands it off, so the URL only decides where the caller's own consent
            lands.
        search (str | Unset): Omit it and the config decides, or search-fast when there is no config.
        start_voice (bool | Unset): Start voice as the session opens, as startSessionVoice does: the agent joins the
            call named in call, else agent:<session id>, which joining creates, and returns once it is there. Left out, the
            conversation is held in writing until voice is started. Default: False.
        sts (str | Unset): A speech-to-speech target. Naming one makes the session native once voice is started: the
            model hears and speaks for itself, so no transcriber, conversation model or voice is opened. Omit it and the
            config decides.
        stt (str | Unset): Omit it and the config decides, or en-low-latency when there is no config.
        tags (CreateSessionRequestTags | Unset): Cost labels, carried onto every request the session makes. Merged with
            the agent config's tags; where both name a key, this one wins.
        timezone (str | Unset): The end user's IANA timezone, such as Europe/Amsterdam, which a device knows and the
            router does not. It sets the zone the agent reads dates in for this session and, for a user the caller may name,
            is remembered as theirs for the scheduled tasks and sessions that follow. An unknown zone is refused with a 400
            invalid_timezone.
        title (str | Unset): What to call the conversation, for a list a person reads, until the router names a
            persistent one for what was said. Never shown to the model: what a conversation is called is a label on it
            rather than part of it.
        tool_timeout_ms (int | Unset): How long the model waits for a tool result. Zero is the default.
        tools (list[SessionTool] | Unset):
        tts (str | Unset): Omit it and the config decides, or en-low-latency when there is no config.
        user_id (str | Unset): Who the agent joins the call as. Default: 'vision-agent'.
        user_name (str | Unset):  Default: 'Vision Agent'.
        video (SessionVideo | Unset):
        voice (str | Unset): The voice to speak in: a provider's own voice id, or one of your voices by id or by the
            name you gave it. Prefix it with custom: to mean only the latter: without the prefix a name that is not one of
            yours is passed through to the provider's library, and with it a name that is not one of yours is refused. A
            native session (sts) takes the vendor's own voice name instead, such as marin at OpenAI.
    """

    agent: str | Unset = UNSET
    agent_id: str | Unset = UNSET
    backchannel: bool | Unset = False
    call: SessionCall | Unset = UNSET
    config_id: str | Unset = UNSET
    connector_bindings: list[SessionConnectorBinding] | Unset = UNSET
    context_truncated: bool | Unset = UNSET
    custom: CreateSessionRequestCustom | Unset = UNSET
    description: str | Unset = UNSET
    greeting: Greeting | Unset = UNSET
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
    resume_history: ResumeHistory | Unset = UNSET
    return_url: str | Unset = UNSET
    search: str | Unset = UNSET
    start_voice: bool | Unset = False
    sts: str | Unset = UNSET
    stt: str | Unset = UNSET
    tags: CreateSessionRequestTags | Unset = UNSET
    timezone: str | Unset = UNSET
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

        call: dict[str, Any] | Unset = UNSET
        if not isinstance(self.call, Unset):
            call = self.call.to_dict()

        config_id = self.config_id

        connector_bindings: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.connector_bindings, Unset):
            connector_bindings = []
            for connector_bindings_item_data in self.connector_bindings:
                connector_bindings_item = connector_bindings_item_data.to_dict()
                connector_bindings.append(connector_bindings_item)

        context_truncated = self.context_truncated

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        description = self.description

        greeting: dict[str, Any] | Unset = UNSET
        if not isinstance(self.greeting, Unset):
            greeting = self.greeting.to_dict()

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

        resume_history: dict[str, Any] | Unset = UNSET
        if not isinstance(self.resume_history, Unset):
            resume_history = self.resume_history.to_dict()

        return_url = self.return_url

        search = self.search

        start_voice = self.start_voice

        sts = self.sts

        stt = self.stt

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        timezone = self.timezone

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
        if call is not UNSET:
            field_dict["call"] = call
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if connector_bindings is not UNSET:
            field_dict["connector_bindings"] = connector_bindings
        if context_truncated is not UNSET:
            field_dict["context_truncated"] = context_truncated
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
        if resume_history is not UNSET:
            field_dict["resume_history"] = resume_history
        if return_url is not UNSET:
            field_dict["return_url"] = return_url
        if search is not UNSET:
            field_dict["search"] = search
        if start_voice is not UNSET:
            field_dict["start_voice"] = start_voice
        if sts is not UNSET:
            field_dict["sts"] = sts
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tags is not UNSET:
            field_dict["tags"] = tags
        if timezone is not UNSET:
            field_dict["timezone"] = timezone
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
        from ..models.create_session_request_custom import CreateSessionRequestCustom
        from ..models.create_session_request_tags import CreateSessionRequestTags
        from ..models.greeting import Greeting
        from ..models.history_message import HistoryMessage
        from ..models.model_overwrites import ModelOverwrites
        from ..models.resume_history import ResumeHistory
        from ..models.session_call import SessionCall
        from ..models.session_connector_binding import SessionConnectorBinding
        from ..models.session_memory import SessionMemory
        from ..models.session_phone import SessionPhone
        from ..models.session_tool import SessionTool
        from ..models.session_video import SessionVideo

        d = dict(src_dict)
        agent = d.pop("agent", UNSET)

        agent_id = d.pop("agent_id", UNSET)

        backchannel = d.pop("backchannel", UNSET)

        _call = d.pop("call", UNSET)
        call: SessionCall | Unset
        if isinstance(_call, Unset):
            call = UNSET
        else:
            call = SessionCall.from_dict(_call)

        config_id = d.pop("config_id", UNSET)

        _connector_bindings = d.pop("connector_bindings", UNSET)
        connector_bindings: list[SessionConnectorBinding] | Unset = UNSET
        if _connector_bindings is not UNSET:
            connector_bindings = []
            for connector_bindings_item_data in _connector_bindings:
                connector_bindings_item = SessionConnectorBinding.from_dict(
                    connector_bindings_item_data
                )

                connector_bindings.append(connector_bindings_item)

        context_truncated = d.pop("context_truncated", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: CreateSessionRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = CreateSessionRequestCustom.from_dict(_custom)

        description = d.pop("description", UNSET)

        _greeting = d.pop("greeting", UNSET)
        greeting: Greeting | Unset
        if isinstance(_greeting, Unset):
            greeting = UNSET
        else:
            greeting = Greeting.from_dict(_greeting)

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

        _resume_history = d.pop("resume_history", UNSET)
        resume_history: ResumeHistory | Unset
        if isinstance(_resume_history, Unset):
            resume_history = UNSET
        else:
            resume_history = ResumeHistory.from_dict(_resume_history)

        return_url = d.pop("return_url", UNSET)

        search = d.pop("search", UNSET)

        start_voice = d.pop("start_voice", UNSET)

        sts = d.pop("sts", UNSET)

        stt = d.pop("stt", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: CreateSessionRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CreateSessionRequestTags.from_dict(_tags)

        timezone = d.pop("timezone", UNSET)

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
            call=call,
            config_id=config_id,
            connector_bindings=connector_bindings,
            context_truncated=context_truncated,
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
            resume_history=resume_history,
            return_url=return_url,
            search=search,
            start_voice=start_voice,
            sts=sts,
            stt=stt,
            tags=tags,
            timezone=timezone,
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
