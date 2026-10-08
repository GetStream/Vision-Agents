from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="PluginClient")


@_attrs_define
class PluginClient:
    """The OAuth client a config logs a plugin in with. Its secret is sealed and never returned.

    Attributes:
        client_id (str): The client id the provider issued.
        has_secret (bool): Whether a client secret is stored. It is never returned.
    """

    client_id: str
    has_secret: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        client_id = self.client_id

        has_secret = self.has_secret

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "client_id": client_id,
                "has_secret": has_secret,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        client_id = d.pop("client_id")

        has_secret = d.pop("has_secret")

        plugin_client = cls(
            client_id=client_id,
            has_secret=has_secret,
        )

        plugin_client.additional_properties = d
        return plugin_client

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
