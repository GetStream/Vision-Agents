from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector import Connector
from ...models.custom_connector_request import CustomConnectorRequest
from ...models.error import Error
from ...types import Response


def _get_kwargs(
    *,
    body: CustomConnectorRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connectors",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Connector | Error | None:
    if response.status_code == 200:
        response_200 = Connector.from_dict(response.json())

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

    if response.status_code == 500:
        response_500 = Error.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Connector | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CustomConnectorRequest,
) -> Response[Connector | Error]:
    """Add a custom MCP connector

     Stores a custom MCP server as one of the app's connectors. Sending an id the app already has stores
    its next revision, which connections pick up when they reconnect; sending the same definition again
    changes nothing. A built-in cannot be changed this way.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomConnectorRequest): A custom MCP server for the app's agents to connect to. An
            unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connector | Error]
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
    body: CustomConnectorRequest,
) -> Connector | Error | None:
    """Add a custom MCP connector

     Stores a custom MCP server as one of the app's connectors. Sending an id the app already has stores
    its next revision, which connections pick up when they reconnect; sending the same definition again
    changes nothing. A built-in cannot be changed this way.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomConnectorRequest): A custom MCP server for the app's agents to connect to. An
            unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connector | Error
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CustomConnectorRequest,
) -> Response[Connector | Error]:
    """Add a custom MCP connector

     Stores a custom MCP server as one of the app's connectors. Sending an id the app already has stores
    its next revision, which connections pick up when they reconnect; sending the same definition again
    changes nothing. A built-in cannot be changed this way.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomConnectorRequest): A custom MCP server for the app's agents to connect to. An
            unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connector | Error]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: CustomConnectorRequest,
) -> Connector | Error | None:
    """Add a custom MCP connector

     Stores a custom MCP server as one of the app's connectors. Sending an id the app already has stores
    its next revision, which connections pick up when they reconnect; sending the same definition again
    changes nothing. A built-in cannot be changed this way.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomConnectorRequest): A custom MCP server for the app's agents to connect to. An
            unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connector | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
