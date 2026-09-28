from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.tier import Tier
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.provider_benchmark import ProviderBenchmark
    from ..models.provider_health import ProviderHealth
    from ..models.provider_price import ProviderPrice


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
        description (str | Unset): What the model is good at, and what that costs in speed or money. Empty if the
            deployment wrote none.
        usage_share (float | Unset): This model's share of the modality's requests over the last seven days, across
            every customer, from 0 to 1. It is how popular the model is, and is 0 when nothing was served or the deployment
            keeps no statistics.
        benchmark (ProviderBenchmark | Unset): What Artificial Analysis measured for this model, refreshed by hand
            rather than live. A field is absent when the model was not measured on it.
        price (ProviderPrice | Unset): What this deployment is billed for the model, in US dollars. A rate is absent
            when the model is not billed by that unit.
    """

    provider: str
    model: str
    languages: list[str]
    realtime: bool
    tier: Tier
    health: ProviderHealth
    description: str | Unset = UNSET
    usage_share: float | Unset = UNSET
    benchmark: ProviderBenchmark | Unset = UNSET
    price: ProviderPrice | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        model = self.model

        languages = self.languages

        realtime = self.realtime

        tier = self.tier.value

        health = self.health.to_dict()

        description = self.description

        usage_share = self.usage_share

        benchmark: dict[str, Any] | Unset = UNSET
        if not isinstance(self.benchmark, Unset):
            benchmark = self.benchmark.to_dict()

        price: dict[str, Any] | Unset = UNSET
        if not isinstance(self.price, Unset):
            price = self.price.to_dict()

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
        if description is not UNSET:
            field_dict["description"] = description
        if usage_share is not UNSET:
            field_dict["usage_share"] = usage_share
        if benchmark is not UNSET:
            field_dict["benchmark"] = benchmark
        if price is not UNSET:
            field_dict["price"] = price

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.provider_benchmark import ProviderBenchmark
        from ..models.provider_health import ProviderHealth
        from ..models.provider_price import ProviderPrice

        d = dict(src_dict)
        provider = d.pop("provider")

        model = d.pop("model")

        languages = cast(list[str], d.pop("languages"))

        realtime = d.pop("realtime")

        tier = Tier(d.pop("tier"))

        health = ProviderHealth.from_dict(d.pop("health"))

        description = d.pop("description", UNSET)

        usage_share = d.pop("usage_share", UNSET)

        _benchmark = d.pop("benchmark", UNSET)
        benchmark: ProviderBenchmark | Unset
        if isinstance(_benchmark, Unset):
            benchmark = UNSET
        else:
            benchmark = ProviderBenchmark.from_dict(_benchmark)

        _price = d.pop("price", UNSET)
        price: ProviderPrice | Unset
        if isinstance(_price, Unset):
            price = UNSET
        else:
            price = ProviderPrice.from_dict(_price)

        provider = cls(
            provider=provider,
            model=model,
            languages=languages,
            realtime=realtime,
            tier=tier,
            health=health,
            description=description,
            usage_share=usage_share,
            benchmark=benchmark,
            price=price,
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
