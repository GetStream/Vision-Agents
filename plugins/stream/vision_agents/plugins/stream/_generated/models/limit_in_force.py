from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.limit_level import LimitLevel

T = TypeVar("T", bound="LimitInForce")


@_attrs_define
class LimitInForce:
    """
    Attributes:
        effective_limit_micros (int):
        id (int):
        level (LimitLevel): Whose ids a limit names. A key's or a user's limit at organization or app level is a default
            for every key or user there, each counted on its own; at api_key or user level it names one. The most specific
            level wins. Every other subject has one level: the organization, a category and a team are organization level;
            an app, and a category or team within one, app level.
        limit_micros (int):
    """

    effective_limit_micros: int
    id: int
    level: LimitLevel
    limit_micros: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        effective_limit_micros = self.effective_limit_micros

        id = self.id

        level = self.level.value

        limit_micros = self.limit_micros

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "effective_limit_micros": effective_limit_micros,
                "id": id,
                "level": level,
                "limit_micros": limit_micros,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        effective_limit_micros = d.pop("effective_limit_micros")

        id = d.pop("id")

        level = LimitLevel(d.pop("level"))

        limit_micros = d.pop("limit_micros")

        limit_in_force = cls(
            effective_limit_micros=effective_limit_micros,
            id=id,
            level=level,
            limit_micros=limit_micros,
        )

        limit_in_force.additional_properties = d
        return limit_in_force

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
