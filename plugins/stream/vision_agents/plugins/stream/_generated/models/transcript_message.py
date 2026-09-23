from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="TranscriptMessage")


@_attrs_define
class TranscriptMessage:
    """
    Attributes:
        speaker (str): Who said it, the agent under its own user id.
        text (str):
        created_at (datetime.datetime):
        name (str | Unset): That speaker's display name, when they have one.
        agent (bool | Unset): Whether the agent said it rather than somebody it was talking to. It is what the line was
            stored as, so it holds however the agent was named.
    """

    speaker: str
    text: str
    created_at: datetime.datetime
    name: str | Unset = UNSET
    agent: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        speaker = self.speaker

        text = self.text

        created_at = self.created_at.isoformat()

        name = self.name

        agent = self.agent

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "speaker": speaker,
                "text": text,
                "created_at": created_at,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if agent is not UNSET:
            field_dict["agent"] = agent

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        speaker = d.pop("speaker")

        text = d.pop("text")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        name = d.pop("name", UNSET)

        agent = d.pop("agent", UNSET)

        transcript_message = cls(
            speaker=speaker,
            text=text,
            created_at=created_at,
            name=name,
            agent=agent,
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
