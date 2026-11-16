from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorProviderAppRequest")


@_attrs_define
class ConnectorProviderAppRequest:
    """The app the router creates and keeps in the customer's workspace. An unknown field is refused rather than ignored.

    Attributes:
        name (str): The app's name in the customer's workspace.
        allowed_ip_address_ranges (list[str] | None | Unset): IP addresses or CIDR ranges the app's tokens work from, at
            most 10. Left out, they work from anywhere.
        config_refresh_token (str | Unset): The refresh token of an app configuration token a workspace admin generated
            in Slack's app settings. Required the first time; the router rotates it and keeps the result sealed. Sent again,
            it replaces the one kept. Never returned.
    """

    name: str
    allowed_ip_address_ranges: list[str] | None | Unset = UNSET
    config_refresh_token: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        allowed_ip_address_ranges: list[str] | None | Unset
        if isinstance(self.allowed_ip_address_ranges, Unset):
            allowed_ip_address_ranges = UNSET
        elif isinstance(self.allowed_ip_address_ranges, list):
            allowed_ip_address_ranges = self.allowed_ip_address_ranges

        else:
            allowed_ip_address_ranges = self.allowed_ip_address_ranges

        config_refresh_token = self.config_refresh_token

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
            }
        )
        if allowed_ip_address_ranges is not UNSET:
            field_dict["allowed_ip_address_ranges"] = allowed_ip_address_ranges
        if config_refresh_token is not UNSET:
            field_dict["config_refresh_token"] = config_refresh_token

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        def _parse_allowed_ip_address_ranges(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                allowed_ip_address_ranges_type_0 = cast(list[str], data)

                return allowed_ip_address_ranges_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        allowed_ip_address_ranges = _parse_allowed_ip_address_ranges(
            d.pop("allowed_ip_address_ranges", UNSET)
        )

        config_refresh_token = d.pop("config_refresh_token", UNSET)

        connector_provider_app_request = cls(
            name=name,
            allowed_ip_address_ranges=allowed_ip_address_ranges,
            config_refresh_token=config_refresh_token,
        )

        return connector_provider_app_request
