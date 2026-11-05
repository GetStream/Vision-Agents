from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_tool_input_schema import ConnectionToolInputSchema


T = TypeVar("T", bound="ConnectionTool")


@_attrs_define
class ConnectionTool:
    """
    Attributes:
        description (str):
        input_schema (ConnectionToolInputSchema): The JSON Schema of its arguments.
        name (str): The tool's name at the provider. An agent config grants it by this name.
        read_only (bool): Whether the tool only reads, as the connector's latest revision says: its source lists it in
            read_only_tools, or says outbound: false. A binding may grant such a tool by name beside a sensitive connector
            without require_approval; any other tool there needs require_approval.
        schema_digest (str): The SHA-256 of its name, description and input schema. A grant pins it, so a tool whose
            schema changes is not offered until it is granted again.
        needs_scopes (list[str] | None | Unset): The scopes a call of the tool needs, as the connector says. Absent when
            it says none.
    """

    description: str
    input_schema: ConnectionToolInputSchema
    name: str
    read_only: bool
    schema_digest: str
    needs_scopes: list[str] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        input_schema = self.input_schema.to_dict()

        name = self.name

        read_only = self.read_only

        schema_digest = self.schema_digest

        needs_scopes: list[str] | None | Unset
        if isinstance(self.needs_scopes, Unset):
            needs_scopes = UNSET
        elif isinstance(self.needs_scopes, list):
            needs_scopes = self.needs_scopes

        else:
            needs_scopes = self.needs_scopes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "description": description,
                "input_schema": input_schema,
                "name": name,
                "read_only": read_only,
                "schema_digest": schema_digest,
            }
        )
        if needs_scopes is not UNSET:
            field_dict["needs_scopes"] = needs_scopes

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connection_tool_input_schema import ConnectionToolInputSchema

        d = dict(src_dict)
        description = d.pop("description")

        input_schema = ConnectionToolInputSchema.from_dict(d.pop("input_schema"))

        name = d.pop("name")

        read_only = d.pop("read_only")

        schema_digest = d.pop("schema_digest")

        def _parse_needs_scopes(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                needs_scopes_type_0 = cast(list[str], data)

                return needs_scopes_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        needs_scopes = _parse_needs_scopes(d.pop("needs_scopes", UNSET))

        connection_tool = cls(
            description=description,
            input_schema=input_schema,
            name=name,
            read_only=read_only,
            schema_digest=schema_digest,
            needs_scopes=needs_scopes,
        )

        connection_tool.additional_properties = d
        return connection_tool

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
