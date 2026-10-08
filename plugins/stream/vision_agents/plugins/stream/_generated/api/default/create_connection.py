from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection import Connection
from ...models.connection_request import ConnectionRequest
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    *,
    body: ConnectionRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connections",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Connection | ErrorResponse | None:
    if response.status_code == 201:
        response_201 = Connection.from_dict(response.json())

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
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionRequest,
) -> Response[Connection | ErrorResponse]:
    """Create a connection

     A pending connection to one account at a connector, made from the connector's newest revision. An
    app-owned connection is the app's, for any of its agents. A user-owned one is the user's the backend
    acts for: owner.user_id must be the user X-Stream-User-Id names. Credentials are added afterwards. A
    deployment with connectors off refuses every create.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConnectionRequest): A connection to create, pending until an account is connected.
            An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionRequest,
) -> Connection | ErrorResponse | None:
    """Create a connection

     A pending connection to one account at a connector, made from the connector's newest revision. An
    app-owned connection is the app's, for any of its agents. A user-owned one is the user's the backend
    acts for: owner.user_id must be the user X-Stream-User-Id names. Credentials are added afterwards. A
    deployment with connectors off refuses every create.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConnectionRequest): A connection to create, pending until an account is connected.
            An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionRequest,
) -> Response[Connection | ErrorResponse]:
    """Create a connection

     A pending connection to one account at a connector, made from the connector's newest revision. An
    app-owned connection is the app's, for any of its agents. A user-owned one is the user's the backend
    acts for: owner.user_id must be the user X-Stream-User-Id names. Credentials are added afterwards. A
    deployment with connectors off refuses every create.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConnectionRequest): A connection to create, pending until an account is connected.
            An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionRequest,
) -> Connection | ErrorResponse | None:
    """Create a connection

     A pending connection to one account at a connector, made from the connector's newest revision. An
    app-owned connection is the app's, for any of its agents. A user-owned one is the user's the backend
    acts for: owner.user_id must be the user X-Stream-User-Id names. Credentials are added afterwards. A
    deployment with connectors off refuses every create.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConnectionRequest): A connection to create, pending until an account is connected.
            An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
