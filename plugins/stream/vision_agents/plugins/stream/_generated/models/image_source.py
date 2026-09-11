from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.image_source_detail import ImageSourceDetail
from ..types import UNSET, Unset

T = TypeVar("T", bound="ImageSource")


@_attrs_define
class ImageSource:
    """
    Attributes:
        url (str): Absolute HTTP(S) URL or base64 image data URI.
        detail (ImageSourceDetail | Unset):
    """

    url: str
    detail: ImageSourceDetail | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        url = self.url

        detail: str | Unset = UNSET
        if not isinstance(self.detail, Unset):
            detail = self.detail.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "url": url,
            }
        )
        if detail is not UNSET:
            field_dict["detail"] = detail

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        url = d.pop("url")

        _detail = d.pop("detail", UNSET)
        detail: ImageSourceDetail | Unset
        if isinstance(_detail, Unset):
            detail = UNSET
        else:
            detail = ImageSourceDetail(_detail)

        image_source = cls(
            url=url,
            detail=detail,
        )

        image_source.additional_properties = d
        return image_source

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
