from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.detection_options import DetectionOptions
    from ..models.detection_request_tags import DetectionRequestTags


T = TypeVar("T", bound="DetectionRequest")


@_attrs_define
class DetectionRequest:
    """
    Attributes:
        image (str): The image, base64. A JPEG or PNG of at most 4 MB and 4096 pixels a side; a video frame scaled to
            1024 wide is plenty.
        options (DetectionOptions | Unset): Where to look and what to keep.
        tags (DetectionRequestTags | Unset):
    """

    image: str
    options: DetectionOptions | Unset = UNSET
    tags: DetectionRequestTags | Unset = UNSET
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
        from ..models.detection_options import DetectionOptions
        from ..models.detection_request_tags import DetectionRequestTags

        d = dict(src_dict)
        image = d.pop("image")

        _options = d.pop("options", UNSET)
        options: DetectionOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = DetectionOptions.from_dict(_options)

        _tags = d.pop("tags", UNSET)
        tags: DetectionRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = DetectionRequestTags.from_dict(_tags)

        detection_request = cls(
            image=image,
            options=options,
            tags=tags,
        )

        detection_request.additional_properties = d
        return detection_request

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
