from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.tier import Tier

if TYPE_CHECKING:
    from ..models.provider_health import ProviderHealth


T = TypeVar("T", bound="Provider")


@_attrs_define
class Provider:
    """
    Attributes:
        provider (str):  Example: elevenlabs.
        model (str):  Example: eleven_flash_v2_5.
        languages (list[str]):
        realtime (bool):
        tier (Tier): What the model optimises for.
        health (ProviderHealth):
    """

    provider: str
    model: str
    languages: list[str]
    realtime: bool
    tier: Tier
    health: ProviderHealth
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        model = self.model

        languages = self.languages

        realtime = self.realtime

        tier = self.tier.value

        health = self.health.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "model": model,
                "languages": languages,
                "realtime": realtime,
                "tier": tier,
                "health": health,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.provider_health import ProviderHealth

        d = dict(src_dict)
        provider = d.pop("provider")

        model = d.pop("model")

        languages = cast(list[str], d.pop("languages"))

        realtime = d.pop("realtime")

        tier = Tier(d.pop("tier"))

        health = ProviderHealth.from_dict(d.pop("health"))

        provider = cls(
            provider=provider,
            model=model,
            languages=languages,
            realtime=realtime,
            tier=tier,
            health=health,
        )

        provider.additional_properties = d
        return provider

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
