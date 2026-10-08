from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.call_direction import CallDirection
from ..models.session_mode import SessionMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.call_tags import CallTags
    from ..models.call_usage import CallUsage


T = TypeVar("T", bound="Call")


@_attrs_define
class Call:
    """
    Attributes:
        agent_id (str): Which agent ran it, and where its transcript is kept.
        call_id (str):
        direction (CallDirection):
        id (str): The session that ran the call, which is what it is held by.
        started_at (datetime.datetime):
        campaign_id (str | Unset):
        config_id (str | Unset):
        contact_id (str | Unset):
        ended_at (datetime.datetime | Unset): Absent while the call is still running.
        from_number (str | Unset):
        instructions (str | Unset): What the agent was told to be on this call.
        llm (str | Unset): The target that held the conversation.
        llm_used (str | Unset): The provider/model that held the conversation.
        mode (SessionMode | Unset): How the session hears and speaks: a transcriber, a conversation model and a voice;
            one speech-to-speech model; or in writing.
        review_notes (str | Unset):
        review_score (int | Unset): How well the agent handled it, from 1 to 5.
        skills (list[str] | Unset): What the fast model could hand to the subagent. The instructions behind each name
            are in the skill registry.
        sts (str | Unset): The speech-to-speech target, for a native call, on the same terms as stt.
        sts_used (str | Unset): The provider/model that held a native call, on the same terms as stt_used.
        stt (str | Unset): The transcription target the call ran with, after a session's overrides were folded into
            whatever config it named. This is what was asked for rather than what each turn resolved to: a shortcut is
            several models and routing fails over between them, so per-turn providers are in the request rows.
        stt_used (str | Unset): The provider/model that transcribed, once routing picked one. Empty until somebody has
            been heard, and the last one that served if routing failed over.
        summary (str | Unset): What a model made of the call, written once it was over.
        tags (CallTags | Unset):
        thinking_llm (str | Unset): The target delegated work ran on. Empty means nothing was delegated, which also
            means the skills below were never offered. A text call names its llm, which runs its skills too.
        thinking_llm_used (str | Unset): The provider/model delegated work ran on. Empty when nothing was handed over,
            or when the thinking target was never reached.
        to_number (str | Unset):
        tts (str | Unset): The voice target, on the same terms as stt.
        tts_used (str | Unset): The provider/model that spoke, on the same terms as stt_used.
        usage (CallUsage | Unset): What the call spent, summed over every request it made. Counted once the call is
            over, so it is absent while one is still running. Requests that failed are included: a model that read the
            prompt and then fell over is still billed for it.
        user_id (str | Unset): Who the agent spoke to, as the client's own token named them. Empty for a call the
            customer's backend opened, and for telephony, where the number is the name.
        voice (str | Unset): The voice the call asked for, in the provider's own terms. Empty means the provider's
            default.
        voice_used (str | Unset): The voice that spoke, which is the provider's default when none was asked for. Known
            only while the call is running.
    """

    agent_id: str
    call_id: str
    direction: CallDirection
    id: str
    started_at: datetime.datetime
    campaign_id: str | Unset = UNSET
    config_id: str | Unset = UNSET
    contact_id: str | Unset = UNSET
    ended_at: datetime.datetime | Unset = UNSET
    from_number: str | Unset = UNSET
    instructions: str | Unset = UNSET
    llm: str | Unset = UNSET
    llm_used: str | Unset = UNSET
    mode: SessionMode | Unset = UNSET
    review_notes: str | Unset = UNSET
    review_score: int | Unset = UNSET
    skills: list[str] | Unset = UNSET
    sts: str | Unset = UNSET
    sts_used: str | Unset = UNSET
    stt: str | Unset = UNSET
    stt_used: str | Unset = UNSET
    summary: str | Unset = UNSET
    tags: CallTags | Unset = UNSET
    thinking_llm: str | Unset = UNSET
    thinking_llm_used: str | Unset = UNSET
    to_number: str | Unset = UNSET
    tts: str | Unset = UNSET
    tts_used: str | Unset = UNSET
    usage: CallUsage | Unset = UNSET
    user_id: str | Unset = UNSET
    voice: str | Unset = UNSET
    voice_used: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent_id = self.agent_id

        call_id = self.call_id

        direction = self.direction.value

        id = self.id

        started_at = self.started_at.isoformat()

        campaign_id = self.campaign_id

        config_id = self.config_id

        contact_id = self.contact_id

        ended_at: str | Unset = UNSET
        if not isinstance(self.ended_at, Unset):
            ended_at = self.ended_at.isoformat()

        from_number = self.from_number

        instructions = self.instructions

        llm = self.llm

        llm_used = self.llm_used

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        review_notes = self.review_notes

        review_score = self.review_score

        skills: list[str] | Unset = UNSET
        if not isinstance(self.skills, Unset):
            skills = self.skills

        sts = self.sts

        sts_used = self.sts_used

        stt = self.stt

        stt_used = self.stt_used

        summary = self.summary

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        thinking_llm = self.thinking_llm

        thinking_llm_used = self.thinking_llm_used

        to_number = self.to_number

        tts = self.tts

        tts_used = self.tts_used

        usage: dict[str, Any] | Unset = UNSET
        if not isinstance(self.usage, Unset):
            usage = self.usage.to_dict()

        user_id = self.user_id

        voice = self.voice

        voice_used = self.voice_used

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "agent_id": agent_id,
                "call_id": call_id,
                "direction": direction,
                "id": id,
                "started_at": started_at,
            }
        )
        if campaign_id is not UNSET:
            field_dict["campaign_id"] = campaign_id
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if contact_id is not UNSET:
            field_dict["contact_id"] = contact_id
        if ended_at is not UNSET:
            field_dict["ended_at"] = ended_at
        if from_number is not UNSET:
            field_dict["from_number"] = from_number
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if llm is not UNSET:
            field_dict["llm"] = llm
        if llm_used is not UNSET:
            field_dict["llm_used"] = llm_used
        if mode is not UNSET:
            field_dict["mode"] = mode
        if review_notes is not UNSET:
            field_dict["review_notes"] = review_notes
        if review_score is not UNSET:
            field_dict["review_score"] = review_score
        if skills is not UNSET:
            field_dict["skills"] = skills
        if sts is not UNSET:
            field_dict["sts"] = sts
        if sts_used is not UNSET:
            field_dict["sts_used"] = sts_used
        if stt is not UNSET:
            field_dict["stt"] = stt
        if stt_used is not UNSET:
            field_dict["stt_used"] = stt_used
        if summary is not UNSET:
            field_dict["summary"] = summary
        if tags is not UNSET:
            field_dict["tags"] = tags
        if thinking_llm is not UNSET:
            field_dict["thinking_llm"] = thinking_llm
        if thinking_llm_used is not UNSET:
            field_dict["thinking_llm_used"] = thinking_llm_used
        if to_number is not UNSET:
            field_dict["to_number"] = to_number
        if tts is not UNSET:
            field_dict["tts"] = tts
        if tts_used is not UNSET:
            field_dict["tts_used"] = tts_used
        if usage is not UNSET:
            field_dict["usage"] = usage
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if voice is not UNSET:
            field_dict["voice"] = voice
        if voice_used is not UNSET:
            field_dict["voice_used"] = voice_used

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.call_tags import CallTags
        from ..models.call_usage import CallUsage

        d = dict(src_dict)
        agent_id = d.pop("agent_id")

        call_id = d.pop("call_id")

        direction = CallDirection(d.pop("direction"))

        id = d.pop("id")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        campaign_id = d.pop("campaign_id", UNSET)

        config_id = d.pop("config_id", UNSET)

        contact_id = d.pop("contact_id", UNSET)

        _ended_at = d.pop("ended_at", UNSET)
        ended_at: datetime.datetime | Unset
        if isinstance(_ended_at, Unset):
            ended_at = UNSET
        else:
            ended_at = datetime.datetime.fromisoformat(_ended_at)

        from_number = d.pop("from_number", UNSET)

        instructions = d.pop("instructions", UNSET)

        llm = d.pop("llm", UNSET)

        llm_used = d.pop("llm_used", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: SessionMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = SessionMode(_mode)

        review_notes = d.pop("review_notes", UNSET)

        review_score = d.pop("review_score", UNSET)

        skills = cast(list[str], d.pop("skills", UNSET))

        sts = d.pop("sts", UNSET)

        sts_used = d.pop("sts_used", UNSET)

        stt = d.pop("stt", UNSET)

        stt_used = d.pop("stt_used", UNSET)

        summary = d.pop("summary", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: CallTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CallTags.from_dict(_tags)

        thinking_llm = d.pop("thinking_llm", UNSET)

        thinking_llm_used = d.pop("thinking_llm_used", UNSET)

        to_number = d.pop("to_number", UNSET)

        tts = d.pop("tts", UNSET)

        tts_used = d.pop("tts_used", UNSET)

        _usage = d.pop("usage", UNSET)
        usage: CallUsage | Unset
        if isinstance(_usage, Unset):
            usage = UNSET
        else:
            usage = CallUsage.from_dict(_usage)

        user_id = d.pop("user_id", UNSET)

        voice = d.pop("voice", UNSET)

        voice_used = d.pop("voice_used", UNSET)

        call = cls(
            agent_id=agent_id,
            call_id=call_id,
            direction=direction,
            id=id,
            started_at=started_at,
            campaign_id=campaign_id,
            config_id=config_id,
            contact_id=contact_id,
            ended_at=ended_at,
            from_number=from_number,
            instructions=instructions,
            llm=llm,
            llm_used=llm_used,
            mode=mode,
            review_notes=review_notes,
            review_score=review_score,
            skills=skills,
            sts=sts,
            sts_used=sts_used,
            stt=stt,
            stt_used=stt_used,
            summary=summary,
            tags=tags,
            thinking_llm=thinking_llm,
            thinking_llm_used=thinking_llm_used,
            to_number=to_number,
            tts=tts,
            tts_used=tts_used,
            usage=usage,
            user_id=user_id,
            voice=voice,
            voice_used=voice_used,
        )

        call.additional_properties = d
        return call

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
