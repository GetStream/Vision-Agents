from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection import Connection
from ...models.connection_credentials import ConnectionCredentials
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: ConnectionCredentials,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/connections/{id}/credentials".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Connection | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = Connection.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Connection | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionCredentials,
) -> Response[Connection | ErrorResponse]:
    """Set a connection's credentials

     Stores the credentials a connection's scheme takes, sealed, and connects it: an API key, a bearer
    token, an OAuth client for client credentials, an OAuth grant the provider already issued, or
    nothing for a connector that needs none. expected_revision must be the connection's revision as last
    read; a connection that moved past it is a 409. The values are never shown again. Who may set them
    is who may read the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionCredentials): Credentials for a connection, under the revision the caller
            last read. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionCredentials,
) -> Connection | ErrorResponse | None:
    """Set a connection's credentials

     Stores the credentials a connection's scheme takes, sealed, and connects it: an API key, a bearer
    token, an OAuth client for client credentials, an OAuth grant the provider already issued, or
    nothing for a connector that needs none. expected_revision must be the connection's revision as last
    read; a connection that moved past it is a 409. The values are never shown again. Who may set them
    is who may read the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionCredentials): Credentials for a connection, under the revision the caller
            last read. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionCredentials,
) -> Response[Connection | ErrorResponse]:
    """Set a connection's credentials

     Stores the credentials a connection's scheme takes, sealed, and connects it: an API key, a bearer
    token, an OAuth client for client credentials, an OAuth grant the provider already issued, or
    nothing for a connector that needs none. expected_revision must be the connection's revision as last
    read; a connection that moved past it is a 409. The values are never shown again. Who may set them
    is who may read the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionCredentials): Credentials for a connection, under the revision the caller
            last read. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionCredentials,
) -> Connection | ErrorResponse | None:
    """Set a connection's credentials

     Stores the credentials a connection's scheme takes, sealed, and connects it: an API key, a bearer
    token, an OAuth client for client credentials, an OAuth grant the provider already issued, or
    nothing for a connector that needs none. expected_revision must be the connection's revision as last
    read; a connection that moved past it is a 409. The values are never shown again. Who may set them
    is who may read the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionCredentials): Credentials for a connection, under the revision the caller
            last read. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
