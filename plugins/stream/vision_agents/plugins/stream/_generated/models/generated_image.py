from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.generated_image_media_type import GeneratedImageMediaType
from ..types import UNSET, Unset

T = TypeVar("T", bound="GeneratedImage")


@_attrs_define
class GeneratedImage:
    """
    Attributes:
        media_type (GeneratedImageMediaType): What the picture is, read off the picture itself rather than the
            provider's label.
        width (int):
        height (int):
        data (str): The picture, base64. Decoded and checked before it was returned, and never more than 10 MiB.
        seed (int | Unset): The seed the provider reports, which draws the same picture again. Absent when it reports
            none.
    """

    media_type: GeneratedImageMediaType
    width: int
    height: int
    data: str
    seed: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        media_type = self.media_type.value

        width = self.width

        height = self.height

        data = self.data

        seed = self.seed

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "media_type": media_type,
                "width": width,
                "height": height,
                "data": data,
            }
        )
        if seed is not UNSET:
            field_dict["seed"] = seed

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        media_type = GeneratedImageMediaType(d.pop("media_type"))

        width = d.pop("width")

        height = d.pop("height")

        data = d.pop("data")

        seed = d.pop("seed", UNSET)

        generated_image = cls(
            media_type=media_type,
            width=width,
            height=height,
            data=data,
            seed=seed,
        )

        generated_image.additional_properties = d
        return generated_image

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
