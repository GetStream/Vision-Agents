from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.limit import Limit
    from ..models.window_spend import WindowSpend


T = TypeVar("T", bound="Team")


@_attrs_define
class Team:
    """One of the organization's teams: one a spend limit names, or one this month's work carried. A team has no members:
    each user's token names its team.

        Attributes:
            apps (list[str]): The apps whose work carried the team this calendar month (UTC), by id.
            limits (list[Limit]): The team's spend limits, across the organization and within each app, by id.
            name (str):
            spent (list[WindowSpend]): What the team spent in the current daily, weekly, biweekly and monthly windows.
    """

    apps: list[str]
    limits: list[Limit]
    name: str
    spent: list[WindowSpend]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        apps = self.apps

        limits = []
        for limits_item_data in self.limits:
            limits_item = limits_item_data.to_dict()
            limits.append(limits_item)

        name = self.name

        spent = []
        for spent_item_data in self.spent:
            spent_item = spent_item_data.to_dict()
            spent.append(spent_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "apps": apps,
                "limits": limits,
                "name": name,
                "spent": spent,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.limit import Limit
        from ..models.window_spend import WindowSpend

        d = dict(src_dict)
        apps = cast(list[str], d.pop("apps"))

        limits = []
        _limits = d.pop("limits")
        for limits_item_data in _limits:
            limits_item = Limit.from_dict(limits_item_data)

            limits.append(limits_item)

        name = d.pop("name")

        spent = []
        _spent = d.pop("spent")
        for spent_item_data in _spent:
            spent_item = WindowSpend.from_dict(spent_item_data)

            spent.append(spent_item)

        team = cls(
            apps=apps,
            limits=limits,
            name=name,
            spent=spent,
        )

        team.additional_properties = d
        return team

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
