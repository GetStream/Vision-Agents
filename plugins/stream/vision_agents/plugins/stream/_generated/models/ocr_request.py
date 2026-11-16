from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.ocr_options import OcrOptions
    from ..models.ocr_request_tags import OcrRequestTags


T = TypeVar("T", bound="OcrRequest")


@_attrs_define
class OcrRequest:
    """
    Attributes:
        image (str): The page, base64. A JPEG or PNG of at most 10 MiB and 10,000 pixels a side; a page drawn at 150 to
            200 dpi reads well.
        options (OcrOptions | Unset): Who reads the page, and what may happen to it afterwards.
        tags (OcrRequestTags | Unset):
    """

    image: str
    options: OcrOptions | Unset = UNSET
    tags: OcrRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        image = self.image

        options: dict[str, Any] | Unset = UNSET
        if not isinstance(self.options, Unset):
            options = self.options.to_dict()

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "image": image,
            }
        )
        if options is not UNSET:
            field_dict["options"] = options
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.ocr_options import OcrOptions
        from ..models.ocr_request_tags import OcrRequestTags

        d = dict(src_dict)
        image = d.pop("image")

        _options = d.pop("options", UNSET)
        options: OcrOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = OcrOptions.from_dict(_options)

        _tags = d.pop("tags", UNSET)
        tags: OcrRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = OcrRequestTags.from_dict(_tags)

        ocr_request = cls(
            image=image,
            options=options,
            tags=tags,
        )

        ocr_request.additional_properties = d
        return ocr_request

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
