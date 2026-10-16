from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ConsentReturnUrls")


@_attrs_define
class ConsentReturnUrls:
    """The pages of the app's own that a connector consent may return to, besides the dashboard. Nothing reads the list
    yet: every consent still finishes on the dashboard.

        Attributes:
            urls (list[str]): Absolute URLs of at most 2048 bytes each, kept exactly as sent. Each is https, or http on a
                loopback host (localhost, 127.0.0.0/8 or [::1]) for local development (RFC 8252 sections 7.3 and 8.3). None
                carries a user name, a password or a fragment (RFC 6749 section 3.1.2), and none is listed twice. Empty when
                none are registered.
    """

    urls: list[str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        urls = self.urls

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "urls": urls,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        urls = cast(list[str], d.pop("urls"))

        consent_return_urls = cls(
            urls=urls,
        )

        consent_return_urls.additional_properties = d
        return consent_return_urls

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
