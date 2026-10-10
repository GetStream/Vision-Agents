from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorToolGrant")


@_attrs_define
class ConnectorToolGrant:
    """One tool a binding allows.

    Attributes:
        name (str): The tool as the connector names it.
        schema_digest (str | Unset): The SHA-256 of the tool's name, description and input schema, as 64 lowercase hex
            characters. A tool whose schema has changed since no longer matches and is not offered. Required on a fixed
            binding. A session binding may leave it out: the first session that opens a person's connection pins the digest
            the provider lists then, later sessions offer the tool only while it still matches, and a reconnect pins again.
    """

    name: str
    schema_digest: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        schema_digest = self.schema_digest

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if schema_digest is not UNSET:
            field_dict["schema_digest"] = schema_digest

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        schema_digest = d.pop("schema_digest", UNSET)

        connector_tool_grant = cls(
            name=name,
            schema_digest=schema_digest,
        )

        connector_tool_grant.additional_properties = d
        return connector_tool_grant

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
