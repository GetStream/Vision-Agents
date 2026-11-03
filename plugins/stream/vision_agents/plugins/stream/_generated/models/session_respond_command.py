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
        text (str):
        type_ (SessionRespondCommandType):
        images (list[ImageSource] | Unset):
        request_id (str | Unset): Generated and sent by the SDKs, one per question, so a retry is answered once.
            Required for personal persistent text conversations, and ignored by a session not kept in Stream Chat. Text only
            when present.
    """

    text: str
    type_: SessionRespondCommandType
    images: list[ImageSource] | Unset = UNSET
    request_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        type_ = self.type_.value

        images: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.images, Unset):
            images = []
            for images_item_data in self.images:
                images_item = images_item_data.to_dict()
                images.append(images_item)

        request_id = self.request_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
                "type": type_,
            }
        )
        if images is not UNSET:
            field_dict["images"] = images
        if request_id is not UNSET:
            field_dict["request_id"] = request_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.image_source import ImageSource

        d = dict(src_dict)
        text = d.pop("text")

        type_ = SessionRespondCommandType(d.pop("type"))

        _images = d.pop("images", UNSET)
        images: list[ImageSource] | Unset = UNSET
        if _images is not UNSET:
            images = []
            for images_item_data in _images:
                images_item = ImageSource.from_dict(images_item_data)

                images.append(images_item)

        request_id = d.pop("request_id", UNSET)

        session_respond_command = cls(
            text=text,
            type_=type_,
            images=images,
            request_id=request_id,
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
