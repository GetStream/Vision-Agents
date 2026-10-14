from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    token: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/plugins/events/{token}".format(
            token=quote(str(token), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = cast(Any, None)
        return response_200

    if response.status_code == 202:
        response_202 = cast(Any, None)
        return response_202

    if response.status_code == 401:
        response_401 = cast(Any, None)
        return response_401

    if response.status_code == 410:
        response_410 = cast(Any, None)
        return response_410

    if response.status_code == 413:
        response_413 = cast(Any, None)
        return response_413

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    token: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Receive a plugin's MCP event

     Deprecated: a connector binding's events are delivered to receiveConnectionEvent. Where a plugin's
    MCP server delivers the events an agent subscribed to, signed with Standard Webhooks. The path is
    unauthenticated because the server is not a customer: the token names the subscription and its
    secret signs each delivery. A verification is answered with its challenge, and an event opens a text
    conversation.

    Args:
        token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        token=token,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    token: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Receive a plugin's MCP event

     Deprecated: a connector binding's events are delivered to receiveConnectionEvent. Where a plugin's
    MCP server delivers the events an agent subscribed to, signed with Standard Webhooks. The path is
    unauthenticated because the server is not a customer: the token names the subscription and its
    secret signs each delivery. A verification is answered with its challenge, and an event opens a text
    conversation.

    Args:
        token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        token=token,
        client=client,
    ).parsed


async def asyncio_detailed(
    token: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Receive a plugin's MCP event

     Deprecated: a connector binding's events are delivered to receiveConnectionEvent. Where a plugin's
    MCP server delivers the events an agent subscribed to, signed with Standard Webhooks. The path is
    unauthenticated because the server is not a customer: the token names the subscription and its
    secret signs each delivery. A verification is answered with its challenge, and an event opens a text
    conversation.

    Args:
        token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        token=token,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    token: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Receive a plugin's MCP event

     Deprecated: a connector binding's events are delivered to receiveConnectionEvent. Where a plugin's
    MCP server delivers the events an agent subscribed to, signed with Standard Webhooks. The path is
    unauthenticated because the server is not a customer: the token names the subscription and its
    secret signs each delivery. A verification is answered with its challenge, and an event opens a text
    conversation.

    Args:
        token (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            token=token,
            client=client,
        )
    ).parsed
