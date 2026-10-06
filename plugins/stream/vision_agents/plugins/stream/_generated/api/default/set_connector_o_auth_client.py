from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_o_auth_client import ConnectorOAuthClient
from ...models.connector_o_auth_client_request import ConnectorOAuthClientRequest
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: ConnectorOAuthClientRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/connectors/{id}/oauth-client".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectorOAuthClient | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ConnectorOAuthClient.from_dict(response.json())

        return response_200

    if response.status_code == 201:
        response_201 = ConnectorOAuthClient.from_dict(response.json())

        return response_201

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
) -> Response[ConnectorOAuthClient | ErrorResponse]:
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
    body: ConnectorOAuthClientRequest,
) -> Response[ConnectorOAuthClient | ErrorResponse]:
    """Set the app's own OAuth client for a connector

     Stores the OAuth client the app registered with the connector's provider, for every consent and
    refresh of the app's connections to it. Putting it again replaces it: a rotated secret is used from
    the next refresh of each connection. A new client_id makes the connections consented with the old
    one need a reconnect, since a refresh token is bound to the client it was issued to (RFC 6749
    section 6). A connector whose client.registration does not list customer refuses it. The secret is
    sealed and never returned.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as github or custom_crm.
        body (ConnectorOAuthClientRequest): The OAuth client the app registered with the
            connector's provider. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorOAuthClient | ErrorResponse]
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
    body: ConnectorOAuthClientRequest,
) -> ConnectorOAuthClient | ErrorResponse | None:
    """Set the app's own OAuth client for a connector

     Stores the OAuth client the app registered with the connector's provider, for every consent and
    refresh of the app's connections to it. Putting it again replaces it: a rotated secret is used from
    the next refresh of each connection. A new client_id makes the connections consented with the old
    one need a reconnect, since a refresh token is bound to the client it was issued to (RFC 6749
    section 6). A connector whose client.registration does not list customer refuses it. The secret is
    sealed and never returned.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as github or custom_crm.
        body (ConnectorOAuthClientRequest): The OAuth client the app registered with the
            connector's provider. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorOAuthClient | ErrorResponse
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
    body: ConnectorOAuthClientRequest,
) -> Response[ConnectorOAuthClient | ErrorResponse]:
    """Set the app's own OAuth client for a connector

     Stores the OAuth client the app registered with the connector's provider, for every consent and
    refresh of the app's connections to it. Putting it again replaces it: a rotated secret is used from
    the next refresh of each connection. A new client_id makes the connections consented with the old
    one need a reconnect, since a refresh token is bound to the client it was issued to (RFC 6749
    section 6). A connector whose client.registration does not list customer refuses it. The secret is
    sealed and never returned.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as github or custom_crm.
        body (ConnectorOAuthClientRequest): The OAuth client the app registered with the
            connector's provider. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorOAuthClient | ErrorResponse]
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
    body: ConnectorOAuthClientRequest,
) -> ConnectorOAuthClient | ErrorResponse | None:
    """Set the app's own OAuth client for a connector

     Stores the OAuth client the app registered with the connector's provider, for every consent and
    refresh of the app's connections to it. Putting it again replaces it: a rotated secret is used from
    the next refresh of each connection. A new client_id makes the connections consented with the old
    one need a reconnect, since a refresh token is bound to the client it was issued to (RFC 6749
    section 6). A connector whose client.registration does not list customer refuses it. The secret is
    sealed and never returned.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as github or custom_crm.
        body (ConnectorOAuthClientRequest): The OAuth client the app registered with the
            connector's provider. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorOAuthClient | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
