from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="CampaignDialDay")


@_attrs_define
class CampaignDialDay:
    """
    Attributes:
        answered (int):
        date (datetime.date): YYYY-MM-DD, in UTC.
        dials (int): Calls placed, answering machines included.
        voicemail (int): Calls an answering machine took.
    """

    answered: int
    date: datetime.date
    dials: int
    voicemail: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        answered = self.answered

        date = self.date.isoformat()

        dials = self.dials

        voicemail = self.voicemail

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "answered": answered,
                "date": date,
                "dials": dials,
                "voicemail": voicemail,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        answered = d.pop("answered")

        date = datetime.date.fromisoformat(d.pop("date"))

        dials = d.pop("dials")

        voicemail = d.pop("voicemail")

        campaign_dial_day = cls(
            answered=answered,
            date=date,
            dials=dials,
            voicemail=voicemail,
        )

        campaign_dial_day.additional_properties = d
        return campaign_dial_day

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
