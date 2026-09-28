from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_connection import ConnectorConnection
from ...models.error import Error
from ...models.list_connector_connections_owner_type import (
    ListConnectorConnectionsOwnerType,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    connector_id: str | Unset = UNSET,
    owner_type: ListConnectorConnectionsOwnerType | Unset = UNSET,
    owner_id: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["connector_id"] = connector_id

    json_owner_type: str | Unset = UNSET
    if not isinstance(owner_type, Unset):
        json_owner_type = owner_type.value

    params["owner_type"] = json_owner_type

    params["owner_id"] = owner_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/connections",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[ConnectorConnection] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = ConnectorConnection.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = Error.from_dict(response.json())

        return response_403

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[ConnectorConnection]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    connector_id: str | Unset = UNSET,
    owner_type: ListConnectorConnectionsOwnerType | Unset = UNSET,
    owner_id: str | Unset = UNSET,
) -> Response[Error | list[ConnectorConnection]]:
    """List reusable account connections

     Returns metadata only. OAuth tokens and imported credentials are never returned. Connections belong
    to the app or to one verified application user.

    Args:
        connector_id (str | Unset):
        owner_type (ListConnectorConnectionsOwnerType | Unset):
        owner_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[ConnectorConnection]]
    """

    kwargs = _get_kwargs(
        connector_id=connector_id,
        owner_type=owner_type,
        owner_id=owner_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    connector_id: str | Unset = UNSET,
    owner_type: ListConnectorConnectionsOwnerType | Unset = UNSET,
    owner_id: str | Unset = UNSET,
) -> Error | list[ConnectorConnection] | None:
    """List reusable account connections

     Returns metadata only. OAuth tokens and imported credentials are never returned. Connections belong
    to the app or to one verified application user.

    Args:
        connector_id (str | Unset):
        owner_type (ListConnectorConnectionsOwnerType | Unset):
        owner_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[ConnectorConnection]
    """

    return sync_detailed(
        client=client,
        connector_id=connector_id,
        owner_type=owner_type,
        owner_id=owner_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    connector_id: str | Unset = UNSET,
    owner_type: ListConnectorConnectionsOwnerType | Unset = UNSET,
    owner_id: str | Unset = UNSET,
) -> Response[Error | list[ConnectorConnection]]:
    """List reusable account connections

     Returns metadata only. OAuth tokens and imported credentials are never returned. Connections belong
    to the app or to one verified application user.

    Args:
        connector_id (str | Unset):
        owner_type (ListConnectorConnectionsOwnerType | Unset):
        owner_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[ConnectorConnection]]
    """

    kwargs = _get_kwargs(
        connector_id=connector_id,
        owner_type=owner_type,
        owner_id=owner_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    connector_id: str | Unset = UNSET,
    owner_type: ListConnectorConnectionsOwnerType | Unset = UNSET,
    owner_id: str | Unset = UNSET,
) -> Error | list[ConnectorConnection] | None:
    """List reusable account connections

     Returns metadata only. OAuth tokens and imported credentials are never returned. Connections belong
    to the app or to one verified application user.

    Args:
        connector_id (str | Unset):
        owner_type (ListConnectorConnectionsOwnerType | Unset):
        owner_id (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[ConnectorConnection]
    """

    return (
        await asyncio_detailed(
            client=client,
            connector_id=connector_id,
            owner_type=owner_type,
            owner_id=owner_id,
        )
    ).parsed
