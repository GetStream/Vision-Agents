from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.session_respond_command_type import SessionRespondCommandType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_source import ImageSource


T = TypeVar("T", bound="SessionRespondCommand")


@_attrs_define
class SessionRespondCommand:
    """
    Attributes:
        type_ (SessionRespondCommandType):
        text (str):
        images (list[ImageSource] | Unset):
    """

    type_: SessionRespondCommandType
    text: str
    images: list[ImageSource] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        text = self.text

        images: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.images, Unset):
            images = []
            for images_item_data in self.images:
                images_item = images_item_data.to_dict()
                images.append(images_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
                "text": text,
            }
        )
        if images is not UNSET:
            field_dict["images"] = images

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.image_source import ImageSource

        d = dict(src_dict)
        type_ = SessionRespondCommandType(d.pop("type"))

        text = d.pop("text")

        _images = d.pop("images", UNSET)
        images: list[ImageSource] | Unset = UNSET
        if _images is not UNSET:
            images = []
            for images_item_data in _images:
                images_item = ImageSource.from_dict(images_item_data)

                images.append(images_item)

        session_respond_command = cls(
            type_=type_,
            text=text,
            images=images,
        )

        session_respond_command.additional_properties = d
        return session_respond_command

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
