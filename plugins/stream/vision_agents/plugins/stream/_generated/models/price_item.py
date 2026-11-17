from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_item_when import PriceItemWhen


T = TypeVar("T", bound="PriceItem")


@_attrs_define
class PriceItem:
    """
    Attributes:
        modality (str): Such as llm, stt, tts, sms or phone.
        model (str):
        provider (str):
        region (str): Where the work runs, such as us or eu: a letter and up to 31 of a-z, 0-9 and hyphens. Work is
            priced only by an entry of its own region, never another region's.
        byok (bool | Unset): Work done with the customer's own key, which the vendor bills the customer for: priced only
            by a BYOK entry, and a BYOK entry prices no other work. Left out of an input, false.
        when (PriceItemWhen | Unset): The item attributes the entry applies to, such as {"service_tier": "flex"}. Absent
            is the item's default entry.
    """

    modality: str
    model: str
    provider: str
    region: str
    byok: bool | Unset = UNSET
    when: PriceItemWhen | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        modality = self.modality

        model = self.model

        provider = self.provider

        region = self.region

        byok = self.byok

        when: dict[str, Any] | Unset = UNSET
        if not isinstance(self.when, Unset):
            when = self.when.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "modality": modality,
                "model": model,
                "provider": provider,
                "region": region,
            }
        )
        if byok is not UNSET:
            field_dict["byok"] = byok
        if when is not UNSET:
            field_dict["when"] = when

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_item_when import PriceItemWhen

        d = dict(src_dict)
        modality = d.pop("modality")

        model = d.pop("model")

        provider = d.pop("provider")

        region = d.pop("region")

        byok = d.pop("byok", UNSET)

        _when = d.pop("when", UNSET)
        when: PriceItemWhen | Unset
        if isinstance(_when, Unset):
            when = UNSET
        else:
            when = PriceItemWhen.from_dict(_when)

        price_item = cls(
            modality=modality,
            model=model,
            provider=provider,
            region=region,
            byok=byok,
            when=when,
        )

        price_item.additional_properties = d
        return price_item

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
