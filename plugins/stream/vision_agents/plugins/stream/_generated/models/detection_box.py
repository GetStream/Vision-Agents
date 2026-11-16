from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="DetectionBox")


@_attrs_define
class DetectionBox:
    """A rectangle as fractions of the image: 0,0 is the top left corner and 1,1 the bottom right.

    Attributes:
        x_max (float):
        x_min (float):
        y_max (float):
        y_min (float):
    """

    x_max: float
    x_min: float
    y_max: float
    y_min: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        x_max = self.x_max

        x_min = self.x_min

        y_max = self.y_max

        y_min = self.y_min

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "x_max": x_max,
                "x_min": x_min,
                "y_max": y_max,
                "y_min": y_min,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        x_max = d.pop("x_max")

        x_min = d.pop("x_min")

        y_max = d.pop("y_max")

        y_min = d.pop("y_min")

        detection_box = cls(
            x_max=x_max,
            x_min=x_min,
            y_max=y_max,
            y_min=y_min,
        )

        detection_box.additional_properties = d
        return detection_box

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
