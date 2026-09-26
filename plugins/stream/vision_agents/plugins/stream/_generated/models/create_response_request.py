from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_source import ImageSource


T = TypeVar("T", bound="CreateResponseRequest")


@_attrs_define
class CreateResponseRequest:
    """
    Attributes:
        text (str): What to answer, as though it had been said.
        images (list[ImageSource] | Unset):
        command_id (str | Unset): Required for personal persistent text conversations, and text only. Reuse this ID and
            identical text for retries; a retry starts no second turn and returns no id.
    """

    text: str
    images: list[ImageSource] | Unset = UNSET
    command_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        images: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.images, Unset):
            images = []
            for images_item_data in self.images:
                images_item = images_item_data.to_dict()
                images.append(images_item)

        command_id = self.command_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
            }
        )
        if images is not UNSET:
            field_dict["images"] = images
        if command_id is not UNSET:
            field_dict["command_id"] = command_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.image_source import ImageSource

        d = dict(src_dict)
        text = d.pop("text")

        _images = d.pop("images", UNSET)
        images: list[ImageSource] | Unset = UNSET
        if _images is not UNSET:
            images = []
            for images_item_data in _images:
                images_item = ImageSource.from_dict(images_item_data)

                images.append(images_item)

        command_id = d.pop("command_id", UNSET)

        create_response_request = cls(
            text=text,
            images=images,
            command_id=command_id,
        )

        create_response_request.additional_properties = d
        return create_response_request

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
