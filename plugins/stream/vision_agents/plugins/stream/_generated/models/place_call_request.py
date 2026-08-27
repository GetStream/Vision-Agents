from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.place_call_request_custom import PlaceCallRequestCustom
    from ..models.place_call_request_headers import PlaceCallRequestHeaders
    from ..models.place_call_request_tags import PlaceCallRequestTags


T = TypeVar("T", bound="PlaceCallRequest")


@_attrs_define
class PlaceCallRequest:
    """
    Attributes:
        from_ (str): One of the customer's own numbers, which is what the person sees.
        to (str):
        call_id (str | Unset): The Stream call the answered leg joins, and so the one the agent has to be in. Omit to
            have one named after this call, since two calls from the same number are two conversations.
        call_type (str | Unset): The Stream call type. Omit for "default".
        ring_timeout_seconds (int | Unset): How long to ring before giving up. Omit to leave the vendor's default, which
            is long enough to reach voicemail. A vendor whose call API cannot express it refuses the call rather than
            ringing for its own default.
        initial_digits (str | Unset): Digits pressed once the person answers, for reaching an extension behind a menu,
            e.g. "ww1234#". w is a short pause and W a long one.
        custom (PlaceCallRequestCustom | Unset): Put on the Stream call, where the agent in it can read it. It is set at
            Stream rather than at the vendor, so every vendor can carry it.
        headers (PlaceCallRequestHeaders | Unset): Carried to the person's leg as custom SIP headers. Only some vendors
            can express these, and one that cannot refuses the call.
        tags (PlaceCallRequestTags | Unset):
    """

    from_: str
    to: str
    call_id: str | Unset = UNSET
    call_type: str | Unset = UNSET
    ring_timeout_seconds: int | Unset = UNSET
    initial_digits: str | Unset = UNSET
    custom: PlaceCallRequestCustom | Unset = UNSET
    headers: PlaceCallRequestHeaders | Unset = UNSET
    tags: PlaceCallRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from_ = self.from_

        to = self.to

        call_id = self.call_id

        call_type = self.call_type

        ring_timeout_seconds = self.ring_timeout_seconds

        initial_digits = self.initial_digits

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        headers: dict[str, Any] | Unset = UNSET
        if not isinstance(self.headers, Unset):
            headers = self.headers.to_dict()

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
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if call_type is not UNSET:
            field_dict["call_type"] = call_type
        if ring_timeout_seconds is not UNSET:
            field_dict["ring_timeout_seconds"] = ring_timeout_seconds
        if initial_digits is not UNSET:
            field_dict["initial_digits"] = initial_digits
        if custom is not UNSET:
            field_dict["custom"] = custom
        if headers is not UNSET:
            field_dict["headers"] = headers
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.place_call_request_custom import PlaceCallRequestCustom
        from ..models.place_call_request_headers import PlaceCallRequestHeaders
        from ..models.place_call_request_tags import PlaceCallRequestTags

        d = dict(src_dict)
        from_ = d.pop("from")

        to = d.pop("to")

        call_id = d.pop("call_id", UNSET)

        call_type = d.pop("call_type", UNSET)

        ring_timeout_seconds = d.pop("ring_timeout_seconds", UNSET)

        initial_digits = d.pop("initial_digits", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: PlaceCallRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = PlaceCallRequestCustom.from_dict(_custom)

        _headers = d.pop("headers", UNSET)
        headers: PlaceCallRequestHeaders | Unset
        if isinstance(_headers, Unset):
            headers = UNSET
        else:
            headers = PlaceCallRequestHeaders.from_dict(_headers)

        _tags = d.pop("tags", UNSET)
        tags: PlaceCallRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = PlaceCallRequestTags.from_dict(_tags)

        place_call_request = cls(
            from_=from_,
            to=to,
            call_id=call_id,
            call_type=call_type,
            ring_timeout_seconds=ring_timeout_seconds,
            initial_digits=initial_digits,
            custom=custom,
            headers=headers,
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
