from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AuditChange")


@_attrs_define
class AuditChange:
    """One field that moved, with what it held before and holds now, each in the shape the resource itself is read in.

    Attributes:
        field (str): The field as the resource's own schema names it.
        after (Any | Unset): What it holds now. Absent when it now holds nothing, which is what a deleted resource's
            fields all do.
        before (Any | Unset): What it held, in the shape the resource is read in. Absent when it held nothing, which is
            what a created resource's fields all did.
    """

    field: str
    after: Any | Unset = UNSET
    before: Any | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field = self.field

        after = self.after

        before = self.before

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "field": field,
            }
        )
        if after is not UNSET:
            field_dict["after"] = after
        if before is not UNSET:
            field_dict["before"] = before

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        field = d.pop("field")

        after = d.pop("after", UNSET)

        before = d.pop("before", UNSET)

        audit_change = cls(
            field=field,
            after=after,
            before=before,
        )

        audit_change.additional_properties = d
        return audit_change

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
