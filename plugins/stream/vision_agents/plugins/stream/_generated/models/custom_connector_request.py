from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_client import ConnectorClient


T = TypeVar("T", bound="CustomConnectorRequest")


@_attrs_define
class CustomConnectorRequest:
    """A custom MCP server for the app's agents to connect to. An unknown field is refused rather than ignored.

    Attributes:
        endpoint (str): The MCP server, over Streamable HTTP: a public https URL without userinfo, query or fragment. An
            address on a private network, loopback or link-local is refused.
        id (str): Starts with custom_, which no built-in does, so a custom definition never stands in for one. Creating
            an id that exists stores the next revision, unless the newest already says the same.
        name (str):
        schemes (list[str] | None): How a connection may authenticate. Each must be a scheme this deployment has.
        category (str | Unset):
        client (ConnectorClient | Unset): Where the OAuth client a connection uses may come from, and how the client
            authenticates at the token endpoint.
        description (str | Unset):
        scopes (list[str] | None | Unset): The scopes a consent asks for, each an RFC 6749 scope token.
    """

    endpoint: str
    id: str
    name: str
    schemes: list[str] | None
    category: str | Unset = UNSET
    client: ConnectorClient | Unset = UNSET
    description: str | Unset = UNSET
    scopes: list[str] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        id = self.id

        name = self.name

        schemes: list[str] | None
        if isinstance(self.schemes, list):
            schemes = self.schemes

        else:
            schemes = self.schemes

        category = self.category

        client: dict[str, Any] | Unset = UNSET
        if not isinstance(self.client, Unset):
            client = self.client.to_dict()

        description = self.description

        scopes: list[str] | None | Unset
        if isinstance(self.scopes, Unset):
            scopes = UNSET
        elif isinstance(self.scopes, list):
            scopes = self.scopes

        else:
            scopes = self.scopes

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "endpoint": endpoint,
                "id": id,
                "name": name,
                "schemes": schemes,
            }
        )
        if category is not UNSET:
            field_dict["category"] = category
        if client is not UNSET:
            field_dict["client"] = client
        if description is not UNSET:
            field_dict["description"] = description
        if scopes is not UNSET:
            field_dict["scopes"] = scopes

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connector_client import ConnectorClient

        d = dict(src_dict)
        endpoint = d.pop("endpoint")

        id = d.pop("id")

        name = d.pop("name")

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

        category = d.pop("category", UNSET)

        _client = d.pop("client", UNSET)
        client: ConnectorClient | Unset
        if isinstance(_client, Unset):
            client = UNSET
        else:
            client = ConnectorClient.from_dict(_client)

        description = d.pop("description", UNSET)

        def _parse_scopes(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                scopes_type_0 = cast(list[str], data)

                return scopes_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        scopes = _parse_scopes(d.pop("scopes", UNSET))

        custom_connector_request = cls(
            endpoint=endpoint,
            id=id,
            name=name,
            schemes=schemes,
            category=category,
            client=client,
            description=description,
            scopes=scopes,
        )

        return custom_connector_request
