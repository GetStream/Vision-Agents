from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.campaign_time_zone_mode import CampaignTimeZoneMode
from ..types import UNSET, Unset

T = TypeVar("T", bound="CampaignTimeZone")


@_attrs_define
class CampaignTimeZone:
    """
    Attributes:
        fallback (str): An IANA zone, used by contact mode.
        mode (CampaignTimeZoneMode): contact reads the hours where each contact is: their time_zone, else their
            number's, else fallback. fixed reads them in fixed.
        fixed (str | Unset): An IANA zone, used by fixed mode.
    """

    fallback: str
    mode: CampaignTimeZoneMode
    fixed: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        fallback = self.fallback

        mode = self.mode.value

        fixed = self.fixed

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "fallback": fallback,
                "mode": mode,
            }
        )
        if fixed is not UNSET:
            field_dict["fixed"] = fixed

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        fallback = d.pop("fallback")

        mode = CampaignTimeZoneMode._decode(d.pop("mode"))

        fixed = d.pop("fixed", UNSET)

        campaign_time_zone = cls(
            fallback=fallback,
            mode=mode,
            fixed=fixed,
        )

        campaign_time_zone.additional_properties = d
        return campaign_time_zone

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
