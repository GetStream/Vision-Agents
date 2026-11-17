from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ProviderKey")


@_attrs_define
class ProviderKey:
    """
    Attributes:
        created_at (datetime.datetime): When a key for the provider in the region was first put.
        key_hint (str): The key's last four characters, so one key can be told from another; empty for a key under 16
            characters. The key itself is never returned. Example: x9Qa.
        provider (str):  Example: anthropic.
        region (str):  Example: us.
        updated_at (datetime.datetime): When the key now kept was put.
    """

    created_at: datetime.datetime
    key_hint: str
    provider: str
    region: str
    updated_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        key_hint = self.key_hint

        provider = self.provider

        region = self.region

        updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "key_hint": key_hint,
                "provider": provider,
                "region": region,
                "updated_at": updated_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        key_hint = d.pop("key_hint")

        provider = d.pop("provider")

        region = d.pop("region")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        provider_key = cls(
            created_at=created_at,
            key_hint=key_hint,
            provider=provider,
            region=region,
            updated_at=updated_at,
        )

        provider_key.additional_properties = d
        return provider_key

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
