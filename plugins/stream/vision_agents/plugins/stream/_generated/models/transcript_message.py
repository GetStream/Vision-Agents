from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.transcript_message_purpose import TranscriptMessagePurpose
from ..types import UNSET, Unset

T = TypeVar("T", bound="TranscriptMessage")


@_attrs_define
class TranscriptMessage:
    """
    Attributes:
        created_at (datetime.datetime):
        id (str): The stored line's id. A line a participant was heard saying has the line_id its session's hearing and
            heard frames named it by; any other line has an id Chat gave it.
        speaker (str): Who said it, the agent under its own user id.
        text (str):
        agent (bool | Unset): Whether the agent said it rather than somebody it was talking to. It is what the line was
            stored as, so it holds however the agent was named.
        generating (bool | Unset): The line is still being written: a reply still streaming, or words still being said,
            stored as they first were. The session's live frames carry the rest.
        interrupted (bool | Unset): The reply was cut off, by the caller talking over it or its voice failing. Its text
            is what it was going to say, of which some or none was heard.
        name (str | Unset): That speaker's display name, when they have one.
        purpose (TranscriptMessagePurpose | Unset): Why the agent said the line unasked: the call's greeting, a check-in
            on a caller gone quiet, or an update on work still running. Absent on an answer.
        turn_id (str | Unset): The turn a reply was written for, as the session's response frames name it.
    """

    created_at: datetime.datetime
    id: str
    speaker: str
    text: str
    agent: bool | Unset = UNSET
    generating: bool | Unset = UNSET
    interrupted: bool | Unset = UNSET
    name: str | Unset = UNSET
    purpose: TranscriptMessagePurpose | Unset = UNSET
    turn_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        id = self.id

        speaker = self.speaker

        text = self.text

        agent = self.agent

        generating = self.generating

        interrupted = self.interrupted

        name = self.name

        purpose: str | Unset = UNSET
        if not isinstance(self.purpose, Unset):
            purpose = self.purpose.value

        turn_id = self.turn_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "id": id,
                "speaker": speaker,
                "text": text,
            }
        )
        if agent is not UNSET:
            field_dict["agent"] = agent
        if generating is not UNSET:
            field_dict["generating"] = generating
        if interrupted is not UNSET:
            field_dict["interrupted"] = interrupted
        if name is not UNSET:
            field_dict["name"] = name
        if purpose is not UNSET:
            field_dict["purpose"] = purpose
        if turn_id is not UNSET:
            field_dict["turn_id"] = turn_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        speaker = d.pop("speaker")

        text = d.pop("text")

        agent = d.pop("agent", UNSET)

        generating = d.pop("generating", UNSET)

        interrupted = d.pop("interrupted", UNSET)

        name = d.pop("name", UNSET)

        _purpose = d.pop("purpose", UNSET)
        purpose: TranscriptMessagePurpose | Unset
        if isinstance(_purpose, Unset):
            purpose = UNSET
        else:
            purpose = TranscriptMessagePurpose(_purpose)

        turn_id = d.pop("turn_id", UNSET)

        transcript_message = cls(
            created_at=created_at,
            id=id,
            speaker=speaker,
            text=text,
            agent=agent,
            generating=generating,
            interrupted=interrupted,
            name=name,
            purpose=purpose,
            turn_id=turn_id,
        )

        transcript_message.additional_properties = d
        return transcript_message

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
