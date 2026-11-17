from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.call_usage import CallUsage


T = TypeVar("T", bound="RunChildren")


@_attrs_define
class RunChildren:
    """What the runs of the child agents it called spent, by the tool each was called by. Absent when it called none."""

    additional_properties: dict[str, CallUsage] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.call_usage import CallUsage

        d = dict(src_dict)
        run_children = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = CallUsage.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        run_children.additional_properties = additional_properties
        return run_children

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> CallUsage:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: CallUsage) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
