from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.campaign_next_end import CampaignNextEnd
from ..types import UNSET, Unset

T = TypeVar("T", bound="CampaignNext")


@_attrs_define
class CampaignNext:
    """
    Attributes:
        end (CampaignNextEnd | Unset): How the contact ends: successful counts as converted.
        step (str | Unset): Reserved for branching to another step, which is refused for now.
    """

    end: CampaignNextEnd | Unset = UNSET
    step: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        end: str | Unset = UNSET
        if not isinstance(self.end, Unset):
            end = self.end.value

        step = self.step

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if end is not UNSET:
            field_dict["end"] = end
        if step is not UNSET:
            field_dict["step"] = step

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _end = d.pop("end", UNSET)
        end: CampaignNextEnd | Unset
        if isinstance(_end, Unset):
            end = UNSET
        else:
            end = CampaignNextEnd._decode(_end)

        step = d.pop("step", UNSET)

        campaign_next = cls(
            end=end,
            step=step,
        )

        campaign_next.additional_properties = d
        return campaign_next

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
