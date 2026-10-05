from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.provider_health import ProviderHealth


T = TypeVar("T", bound="Candidate")


@_attrs_define
class Candidate:
    """
    Attributes:
        health (ProviderHealth):
        model (str):
        provider (str):
    """

    health: ProviderHealth
    model: str
    provider: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        health = self.health.to_dict()

        model = self.model

        provider = self.provider

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "health": health,
                "model": model,
                "provider": provider,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.provider_health import ProviderHealth

        d = dict(src_dict)
        health = ProviderHealth.from_dict(d.pop("health"))

        model = d.pop("model")

        provider = d.pop("provider")

        candidate = cls(
            health=health,
            model=model,
            provider=provider,
        )

        candidate.additional_properties = d
        return candidate

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
