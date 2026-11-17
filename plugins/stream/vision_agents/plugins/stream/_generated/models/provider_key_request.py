from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ProviderKeyRequest")


@_attrs_define
class ProviderKeyRequest:
    """
    Attributes:
        api_key (str): The key, as the vendor issued it. Stored sealed and never returned.
        provider (str): The vendor the key is for, as the router names it: a language model vendor whose models take a
            key of their own here (openai, anthropic, gemini, xai, deepseek where routed), turbopuffer or daytona. Example:
            anthropic.
        region (str | Unset): The region whose work the key is used for: only the instances running there are tried on
            it. Omitted is us. Example: eu.
    """

    api_key: str
    provider: str
    region: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        api_key = self.api_key

        provider = self.provider

        region = self.region

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "api_key": api_key,
                "provider": provider,
            }
        )
        if region is not UNSET:
            field_dict["region"] = region

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        api_key = d.pop("api_key")

        provider = d.pop("provider")

        region = d.pop("region", UNSET)

        provider_key_request = cls(
            api_key=api_key,
            provider=provider,
            region=region,
        )

        provider_key_request.additional_properties = d
        return provider_key_request

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
