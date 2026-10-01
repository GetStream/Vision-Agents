from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.create_connector_definition_request_auth_mode import (
    CreateConnectorDefinitionRequestAuthMode,
)
from ..types import UNSET, Unset

T = TypeVar("T", bound="CreateConnectorDefinitionRequest")


@_attrs_define
class CreateConnectorDefinitionRequest:
    """
    Attributes:
        id (str):
        name (str):
        endpoint (str): Public HTTPS Streamable HTTP MCP endpoint; query strings are not allowed.
        auth_mode (CreateConnectorDefinitionRequestAuthMode):
        category (str | Unset):
        description (str | Unset):
        api_key_header (str | Unset): Required only for api_key; fixed to this definition.
    """

    id: str
    name: str
    endpoint: str
    auth_mode: CreateConnectorDefinitionRequestAuthMode
    category: str | Unset = UNSET
    description: str | Unset = UNSET
    api_key_header: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        endpoint = self.endpoint

        auth_mode = self.auth_mode.value

        category = self.category

        description = self.description

        api_key_header = self.api_key_header

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "endpoint": endpoint,
                "auth_mode": auth_mode,
            }
        )
        if category is not UNSET:
            field_dict["category"] = category
        if description is not UNSET:
            field_dict["description"] = description
        if api_key_header is not UNSET:
            field_dict["api_key_header"] = api_key_header

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        endpoint = d.pop("endpoint")

        auth_mode = CreateConnectorDefinitionRequestAuthMode(d.pop("auth_mode"))

        category = d.pop("category", UNSET)

        description = d.pop("description", UNSET)

        api_key_header = d.pop("api_key_header", UNSET)

        create_connector_definition_request = cls(
            id=id,
            name=name,
            endpoint=endpoint,
            auth_mode=auth_mode,
            category=category,
            description=description,
            api_key_header=api_key_header,
        )

        create_connector_definition_request.additional_properties = d
        return create_connector_definition_request

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
