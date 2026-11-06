from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.campaign_step import CampaignStep


T = TypeVar("T", bound="CampaignSequence")


@_attrs_define
class CampaignSequence:
    """
    Attributes:
        double_dial (bool): Ring again at once when the first call of an attempt is not answered.
        steps (list[CampaignStep]): Exactly one step for now.
    """

    double_dial: bool
    steps: list[CampaignStep]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        double_dial = self.double_dial

        steps = []
        for steps_item_data in self.steps:
            steps_item = steps_item_data.to_dict()
            steps.append(steps_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "double_dial": double_dial,
                "steps": steps,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.campaign_step import CampaignStep

        d = dict(src_dict)
        double_dial = d.pop("double_dial")

        steps = []
        _steps = d.pop("steps")
        for steps_item_data in _steps:
            steps_item = CampaignStep.from_dict(steps_item_data)

            steps.append(steps_item)

        campaign_sequence = cls(
            double_dial=double_dial,
            steps=steps,
        )

        campaign_sequence.additional_properties = d
        return campaign_sequence

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
