from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.campaign_next import CampaignNext


T = TypeVar("T", bound="CampaignOutcome")


@_attrs_define
class CampaignOutcome:
    """
    Attributes:
        description (str): What the call has to have been for this outcome, which the model reading the transcript goes
            by.
        id (str): Lowercase letters, digits and underscores, starting with a letter.
        name (str):
        next_ (CampaignNext):
    """

    description: str
    id: str
    name: str
    next_: CampaignNext
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        id = self.id

        name = self.name

        next_ = self.next_.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "description": description,
                "id": id,
                "name": name,
                "next": next_,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.campaign_next import CampaignNext

        d = dict(src_dict)
        description = d.pop("description")

        id = d.pop("id")

        name = d.pop("name")

        next_ = CampaignNext.from_dict(d.pop("next"))

        campaign_outcome = cls(
            description=description,
            id=id,
            name=name,
            next_=next_,
        )

        campaign_outcome.additional_properties = d
        return campaign_outcome

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
