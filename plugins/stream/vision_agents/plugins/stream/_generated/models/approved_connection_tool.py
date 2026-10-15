from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ApprovedConnectionTool")


@_attrs_define
class ApprovedConnectionTool:
    """
    Attributes:
        name (str):
        schema_digest (str): The digest the tool is pinned at now.
        previous_schema_digest (str | Unset): The digest it was pinned at before. Absent when it was not pinned for the
            connection's grant.
    """

    name: str
    schema_digest: str
    previous_schema_digest: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        schema_digest = self.schema_digest

        previous_schema_digest = self.previous_schema_digest

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "schema_digest": schema_digest,
            }
        )
        if previous_schema_digest is not UNSET:
            field_dict["previous_schema_digest"] = previous_schema_digest

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        schema_digest = d.pop("schema_digest")

        previous_schema_digest = d.pop("previous_schema_digest", UNSET)

        approved_connection_tool = cls(
            name=name,
            schema_digest=schema_digest,
            previous_schema_digest=previous_schema_digest,
        )

        approved_connection_tool.additional_properties = d
        return approved_connection_tool

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
