from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="TimelineEntry")


@_attrs_define
class TimelineEntry:
    """
    Attributes:
        turn_id (str):
        started_at (datetime.datetime):
        heard (str | Unset): What the caller said, when it can be matched to this exchange.
        said (str | Unset): What the agent answered.
        roundtrip_ms (float | Unset): How long the caller waited between finishing and being answered.
        audio_out_ms (float | Unset): How much the agent spoke.
        interrupted (bool | Unset): Whether the caller talked over the answer.
    """

    turn_id: str
    started_at: datetime.datetime
    heard: str | Unset = UNSET
    said: str | Unset = UNSET
    roundtrip_ms: float | Unset = UNSET
    audio_out_ms: float | Unset = UNSET
    interrupted: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        turn_id = self.turn_id

        started_at = self.started_at.isoformat()

        heard = self.heard

        said = self.said

        roundtrip_ms = self.roundtrip_ms

        audio_out_ms = self.audio_out_ms

        interrupted = self.interrupted

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "turn_id": turn_id,
                "started_at": started_at,
            }
        )
        if heard is not UNSET:
            field_dict["heard"] = heard
        if said is not UNSET:
            field_dict["said"] = said
        if roundtrip_ms is not UNSET:
            field_dict["roundtrip_ms"] = roundtrip_ms
        if audio_out_ms is not UNSET:
            field_dict["audio_out_ms"] = audio_out_ms
        if interrupted is not UNSET:
            field_dict["interrupted"] = interrupted

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        turn_id = d.pop("turn_id")

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        heard = d.pop("heard", UNSET)

        said = d.pop("said", UNSET)

        roundtrip_ms = d.pop("roundtrip_ms", UNSET)

        audio_out_ms = d.pop("audio_out_ms", UNSET)

        interrupted = d.pop("interrupted", UNSET)

        timeline_entry = cls(
            turn_id=turn_id,
            started_at=started_at,
            heard=heard,
            said=said,
            roundtrip_ms=roundtrip_ms,
            audio_out_ms=audio_out_ms,
            interrupted=interrupted,
        )

        timeline_entry.additional_properties = d
        return timeline_entry

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
