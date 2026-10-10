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
        custom (PlaceCallRequestCustom | Unset): Put on the Stream call, where the agent in it can read it. It is set at
            Stream rather than at the vendor, so every vendor can carry it.
        headers (PlaceCallRequestHeaders | Unset): Carried to the person's leg as custom SIP headers. Only some vendors
            can express these, and one that cannot refuses the call.
        initial_digits (str | Unset): Digits pressed once the person answers, for reaching an extension behind a menu,
            e.g. "ww1234#". w is a short pause and W a long one.
        ring_timeout_seconds (int | Unset): How long to ring before giving up. Omit to leave the vendor's default, which
            is long enough to reach voicemail. A vendor whose call API cannot express it refuses the call rather than
            ringing for its own default.
        session_id (str | Unset): The session that holds the call: the answered leg is routed into its call,
            agent:<session id>. It takes what a session id does: up to 64 letters, digits, - and _. Omit to have one chosen,
            since two calls from the same number are two conversations. Open the session under this id with start_voice.
        tags (PlaceCallRequestTags | Unset):
    """

    from_: str
    to: str
    custom: PlaceCallRequestCustom | Unset = UNSET
    headers: PlaceCallRequestHeaders | Unset = UNSET
    initial_digits: str | Unset = UNSET
    ring_timeout_seconds: int | Unset = UNSET
    session_id: str | Unset = UNSET
    tags: PlaceCallRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from_ = self.from_

        to = self.to

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        headers: dict[str, Any] | Unset = UNSET
        if not isinstance(self.headers, Unset):
            headers = self.headers.to_dict()

        initial_digits = self.initial_digits

        ring_timeout_seconds = self.ring_timeout_seconds

        session_id = self.session_id

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
        if custom is not UNSET:
            field_dict["custom"] = custom
        if headers is not UNSET:
            field_dict["headers"] = headers
        if initial_digits is not UNSET:
            field_dict["initial_digits"] = initial_digits
        if ring_timeout_seconds is not UNSET:
            field_dict["ring_timeout_seconds"] = ring_timeout_seconds
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
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

        initial_digits = d.pop("initial_digits", UNSET)

        ring_timeout_seconds = d.pop("ring_timeout_seconds", UNSET)

        session_id = d.pop("session_id", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: PlaceCallRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = PlaceCallRequestTags.from_dict(_tags)

        place_call_request = cls(
            from_=from_,
            to=to,
            custom=custom,
            headers=headers,
            initial_digits=initial_digits,
            ring_timeout_seconds=ring_timeout_seconds,
            session_id=session_id,
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
