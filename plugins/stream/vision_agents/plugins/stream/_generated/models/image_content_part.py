from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.image_content_part_type import ImageContentPartType

if TYPE_CHECKING:
    from ..models.image_source import ImageSource


T = TypeVar("T", bound="ImageContentPart")


@_attrs_define
class ImageContentPart:
    """
    Attributes:
        image_url (ImageSource):
        type_ (ImageContentPartType):
    """

    image_url: ImageSource
    type_: ImageContentPartType
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        image_url = self.image_url.to_dict()

        type_ = self.type_.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "image_url": image_url,
                "type": type_,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.image_source import ImageSource

        d = dict(src_dict)
        image_url = ImageSource.from_dict(d.pop("image_url"))

        type_ = ImageContentPartType(d.pop("type"))

        image_content_part = cls(
            image_url=image_url,
            type_=type_,
        )

        image_content_part.additional_properties = d
        return image_content_part

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
