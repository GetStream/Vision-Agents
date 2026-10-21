from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AuthorizationRequest")


@_attrs_define
class AuthorizationRequest:
    """
    Attributes:
        return_url (str | Unset): The app's own page the browser goes back to when the consent ends, with connection_id
            and status added to its query and its own query kept: the query comes back re-encoded with its keys sorted
            (?a=1&a=2&b becomes ?a=1&a=2&b=&connection_id=...&status=...), and a connection_id or status already in it is
            replaced by the router's. The launch page then takes the handoff only from a page of this URL's origin, which
            opens launch_url. Absent or empty, the consent goes back to the dashboard. An absolute URL of at most 2048
            bytes, whose host is ASCII (an internationalized one in punycode, xn--) and not an IPv4-mapped IPv6 address,
            since a browser's origin would not match it: https, or http on a loopback host (localhost, 127.0.0.0/8 or [::1])
            for local development (RFC 8252 sections 7.3 and 8.3), with no user name or password, no fragment (RFC 6749
            section 3.1.2) and a query of key=value pairs. A native app may instead name its own private-use scheme, in
            reverse domain form (RFC 8252 section 7.1), such as com.example.app:/connected, which needs no host: its launch
            page then takes handoff_token from its own fragment, and the callback hands the browser back to the app.
    """

    return_url: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return_url = self.return_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if return_url is not UNSET:
            field_dict["return_url"] = return_url

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        return_url = d.pop("return_url", UNSET)

        authorization_request = cls(
            return_url=return_url,
        )

        authorization_request.additional_properties = d
        return authorization_request

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
