from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_definition_auth_mode import ConnectorDefinitionAuthMode
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorDefinition")


@_attrs_define
class ConnectorDefinition:
    """A built-in or app-defined remote MCP connector definition.

    Attributes:
        id (str):
        name (str):
        category (str):
        description (str):
        endpoint (str):
        auth_mode (ConnectorDefinitionAuthMode):
        api_key_header (str | Unset): Fixed header name for api_key auth; never supplied by the model.
        scopes (list[str] | Unset):
        instance_required (bool | Unset):
        instance_hint (str | Unset):
    """

    id: str
    name: str
    category: str
    description: str
    endpoint: str
    auth_mode: ConnectorDefinitionAuthMode
    api_key_header: str | Unset = UNSET
    scopes: list[str] | Unset = UNSET
    instance_required: bool | Unset = UNSET
    instance_hint: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        category = self.category

        description = self.description

        endpoint = self.endpoint

        auth_mode = self.auth_mode.value

        api_key_header = self.api_key_header

        scopes: list[str] | Unset = UNSET
        if not isinstance(self.scopes, Unset):
            scopes = self.scopes

        instance_required = self.instance_required

        instance_hint = self.instance_hint

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "category": category,
                "description": description,
                "endpoint": endpoint,
                "auth_mode": auth_mode,
            }
        )
        if api_key_header is not UNSET:
            field_dict["api_key_header"] = api_key_header
        if scopes is not UNSET:
            field_dict["scopes"] = scopes
        if instance_required is not UNSET:
            field_dict["instance_required"] = instance_required
        if instance_hint is not UNSET:
            field_dict["instance_hint"] = instance_hint

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        category = d.pop("category")

        description = d.pop("description")

        endpoint = d.pop("endpoint")

        auth_mode = ConnectorDefinitionAuthMode(d.pop("auth_mode"))

        api_key_header = d.pop("api_key_header", UNSET)

        scopes = cast(list[str], d.pop("scopes", UNSET))

        instance_required = d.pop("instance_required", UNSET)

        instance_hint = d.pop("instance_hint", UNSET)

        connector_definition = cls(
            id=id,
            name=name,
            category=category,
            description=description,
            endpoint=endpoint,
            auth_mode=auth_mode,
            api_key_header=api_key_header,
            scopes=scopes,
            instance_required=instance_required,
            instance_hint=instance_hint,
        )

        connector_definition.additional_properties = d
        return connector_definition

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
