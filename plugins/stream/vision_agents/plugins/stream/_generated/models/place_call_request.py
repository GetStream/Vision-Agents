from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.place_call_request_tags import PlaceCallRequestTags


T = TypeVar("T", bound="PlaceCallRequest")


@_attrs_define
class PlaceCallRequest:
    """
    Attributes:
        from_ (str): One of the customer's own numbers, which is what the person sees.
        to (str):
        sip_uri (str | Unset): The trunk the answered call joins. Omit to have one created, which is what a one-off call
            wants.
        tags (PlaceCallRequestTags | Unset):
    """

    from_: str
    to: str
    sip_uri: str | Unset = UNSET
    tags: PlaceCallRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from_ = self.from_

        to = self.to

        sip_uri = self.sip_uri

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "from": from_,
                "to": to,
            }
        )
        if sip_uri is not UNSET:
            field_dict["sip_uri"] = sip_uri
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.place_call_request_tags import PlaceCallRequestTags

        d = dict(src_dict)
        from_ = d.pop("from")

        to = d.pop("to")

        sip_uri = d.pop("sip_uri", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: PlaceCallRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = PlaceCallRequestTags.from_dict(_tags)

        place_call_request = cls(
            from_=from_,
            to=to,
            sip_uri=sip_uri,
            tags=tags,
        )

        place_call_request.additional_properties = d
        return place_call_request

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
