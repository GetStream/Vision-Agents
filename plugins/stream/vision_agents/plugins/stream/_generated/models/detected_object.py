from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.detection_box import DetectionBox


T = TypeVar("T", bound="DetectedObject")


@_attrs_define
class DetectedObject:
    """
    Attributes:
        box (DetectionBox): A rectangle as fractions of the image: 0,0 is the top left corner and 1,1 the bottom right.
        confidence (float): The model's score from 0 to 1. A model that scores nothing, such as Moondream, reports 1.
        label (str):  Example: person.
        embedding (list[float] | Unset): A vector describing what is in the box. Present only when embeddings was asked
            for.
    """

    box: DetectionBox
    confidence: float
    label: str
    embedding: list[float] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        box = self.box.to_dict()

        confidence = self.confidence

        label = self.label

        embedding: list[float] | Unset = UNSET
        if not isinstance(self.embedding, Unset):
            embedding = self.embedding

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "box": box,
                "confidence": confidence,
                "label": label,
            }
        )
        if embedding is not UNSET:
            field_dict["embedding"] = embedding

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.detection_box import DetectionBox

        d = dict(src_dict)
        box = DetectionBox.from_dict(d.pop("box"))

        confidence = d.pop("confidence")

        label = d.pop("label")

        embedding = cast(list[float], d.pop("embedding", UNSET))

        detected_object = cls(
            box=box,
            confidence=confidence,
            label=label,
            embedding=embedding,
        )

        detected_object.additional_properties = d
        return detected_object

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
