from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

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
        can_record (bool | Unset): Whether RecordSpeech can use this candidate. Only set for text to speech: most of its
            vendors only stream, and a recording passes over them.
    """

    health: ProviderHealth
    model: str
    provider: str
    can_record: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        health = self.health.to_dict()

        model = self.model

        provider = self.provider

        can_record = self.can_record

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "health": health,
                "model": model,
                "provider": provider,
            }
        )
        if can_record is not UNSET:
            field_dict["can_record"] = can_record

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.provider_health import ProviderHealth

        d = dict(src_dict)
        health = ProviderHealth.from_dict(d.pop("health"))

        model = d.pop("model")

        provider = d.pop("provider")

        can_record = d.pop("can_record", UNSET)

        candidate = cls(
            health=health,
            model=model,
            provider=provider,
            can_record=can_record,
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
