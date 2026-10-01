from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_connection import ConnectorConnection
from ...models.error import Error
from ...models.put_connector_credentials_request import PutConnectorCredentialsRequest
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: PutConnectorCredentialsRequest,
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
) -> ConnectorConnection | Error | None:
    if response.status_code == 200:
        response_200 = ConnectorConnection.from_dict(response.json())

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

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if response.status_code == 409:
        response_409 = Error.from_dict(response.json())

        return response_409

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ConnectorConnection | Error]:
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
    body: PutConnectorCredentialsRequest,
) -> Response[ConnectorConnection | Error]:
    """Write a connection credential or import an OAuth grant

     Writes a bearer token or API key encrypted at rest, imports a provider-issued OAuth grant, or
    activates a no-auth connection. OAuth endpoints, resource identity, and client authentication are
    resolved from the connector definition or trusted provider metadata. Credential material is write-
    only and updates require the current connection revision.

    Args:
        id (str):
        body (PutConnectorCredentialsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorConnection | Error]
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
    body: PutConnectorCredentialsRequest,
) -> ConnectorConnection | Error | None:
    """Write a connection credential or import an OAuth grant

     Writes a bearer token or API key encrypted at rest, imports a provider-issued OAuth grant, or
    activates a no-auth connection. OAuth endpoints, resource identity, and client authentication are
    resolved from the connector definition or trusted provider metadata. Credential material is write-
    only and updates require the current connection revision.

    Args:
        id (str):
        body (PutConnectorCredentialsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorConnection | Error
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
    body: PutConnectorCredentialsRequest,
) -> Response[ConnectorConnection | Error]:
    """Write a connection credential or import an OAuth grant

     Writes a bearer token or API key encrypted at rest, imports a provider-issued OAuth grant, or
    activates a no-auth connection. OAuth endpoints, resource identity, and client authentication are
    resolved from the connector definition or trusted provider metadata. Credential material is write-
    only and updates require the current connection revision.

    Args:
        id (str):
        body (PutConnectorCredentialsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorConnection | Error]
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
    body: PutConnectorCredentialsRequest,
) -> ConnectorConnection | Error | None:
    """Write a connection credential or import an OAuth grant

     Writes a bearer token or API key encrypted at rest, imports a provider-issued OAuth grant, or
    activates a no-auth connection. OAuth endpoints, resource identity, and client authentication are
    resolved from the connector definition or trusted provider metadata. Credential material is write-
    only and updates require the current connection revision.

    Args:
        id (str):
        body (PutConnectorCredentialsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorConnection | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
