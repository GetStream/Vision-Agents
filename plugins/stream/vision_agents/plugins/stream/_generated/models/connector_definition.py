from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_client import ConnectorClient
    from ..models.connector_input import ConnectorInput


T = TypeVar("T", bound="ConnectorDefinition")


@_attrs_define
class ConnectorDefinition:
    """A connector: an account elsewhere an agent may reach, built in or the app's own. Only what a caller chooses between
    is shown. Endpoints, how an account is recognised, refresh and rate limits stay with the router.

        Attributes:
            client (ConnectorClient): Who may own the OAuth client a connection uses, and how the client authenticates at
                the token endpoint.
            created_at (datetime.datetime): When this revision was stored.
            custom (bool): The app's own definition rather than a built-in.
            id (str): Unique among the built-ins and the app's own. A custom definition's starts with custom_, and a built-
                in's never does.
            inputs (list[ConnectorInput] | None): What a connection is created with, such as a region or a shop.
            name (str):
            revision (int): The manifest's revision. A connection is created from the newest one and keeps reading it until
                it is reconnected.
            schemes (list[str] | None): How a connection may authenticate, such as oauth2_code.
            scopes (list[str] | None): The scopes a consent asks for.
            category (str | Unset):
            description (str | Unset):
    """

    client: ConnectorClient
    created_at: datetime.datetime
    custom: bool
    id: str
    inputs: list[ConnectorInput] | None
    name: str
    revision: int
    schemes: list[str] | None
    scopes: list[str] | None
    category: str | Unset = UNSET
    description: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        client = self.client.to_dict()

        created_at = self.created_at.isoformat()

        custom = self.custom

        id = self.id

        inputs: list[dict[str, Any]] | None
        if isinstance(self.inputs, list):
            inputs = []
            for inputs_type_0_item_data in self.inputs:
                inputs_type_0_item = inputs_type_0_item_data.to_dict()
                inputs.append(inputs_type_0_item)

        else:
            inputs = self.inputs

        name = self.name

        revision = self.revision

        schemes: list[str] | None
        if isinstance(self.schemes, list):
            schemes = self.schemes

        else:
            schemes = self.schemes

        scopes: list[str] | None
        if isinstance(self.scopes, list):
            scopes = self.scopes

        else:
            scopes = self.scopes

        category = self.category

        description = self.description

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "client": client,
                "created_at": created_at,
                "custom": custom,
                "id": id,
                "inputs": inputs,
                "name": name,
                "revision": revision,
                "schemes": schemes,
                "scopes": scopes,
            }
        )
        if category is not UNSET:
            field_dict["category"] = category
        if description is not UNSET:
            field_dict["description"] = description

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connector_client import ConnectorClient
        from ..models.connector_input import ConnectorInput

        d = dict(src_dict)
        client = ConnectorClient.from_dict(d.pop("client"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        custom = d.pop("custom")

        id = d.pop("id")

        def _parse_inputs(data: object) -> list[ConnectorInput] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                inputs_type_0 = []
                _inputs_type_0 = data
                for inputs_type_0_item_data in _inputs_type_0:
                    inputs_type_0_item = ConnectorInput.from_dict(
                        inputs_type_0_item_data
                    )

                    inputs_type_0.append(inputs_type_0_item)

                return inputs_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ConnectorInput] | None, data)

        inputs = _parse_inputs(d.pop("inputs"))

        name = d.pop("name")

        revision = d.pop("revision")

        def _parse_schemes(data: object) -> list[str] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                schemes_type_0 = cast(list[str], data)

                return schemes_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None, data)

        schemes = _parse_schemes(d.pop("schemes"))

        def _parse_scopes(data: object) -> list[str] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                scopes_type_0 = cast(list[str], data)

                return scopes_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None, data)

        scopes = _parse_scopes(d.pop("scopes"))

        category = d.pop("category", UNSET)

        description = d.pop("description", UNSET)

        connector_definition = cls(
            client=client,
            created_at=created_at,
            custom=custom,
            id=id,
            inputs=inputs,
            name=name,
            revision=revision,
            schemes=schemes,
            scopes=scopes,
            category=category,
            description=description,
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
