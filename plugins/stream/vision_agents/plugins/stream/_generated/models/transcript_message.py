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
        created_at (datetime.datetime):
        speaker (str): Who said it, the agent under its own user id.
        text (str):
        agent (bool | Unset): Whether the agent said it rather than somebody it was talking to. It is what the line was
            stored as, so it holds however the agent was named.
        name (str | Unset): That speaker's display name, when they have one.
    """

    created_at: datetime.datetime
    speaker: str
    text: str
    agent: bool | Unset = UNSET
    name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        speaker = self.speaker

        text = self.text

        agent = self.agent

        name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "speaker": speaker,
                "text": text,
            }
        )
        if agent is not UNSET:
            field_dict["agent"] = agent
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        speaker = d.pop("speaker")

        text = d.pop("text")

        agent = d.pop("agent", UNSET)

        name = d.pop("name", UNSET)

        transcript_message = cls(
            created_at=created_at,
            speaker=speaker,
            text=text,
            agent=agent,
            name=name,
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
