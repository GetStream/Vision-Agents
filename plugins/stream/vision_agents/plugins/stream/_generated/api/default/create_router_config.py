from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.router_config import RouterConfig
from ...models.router_config_request import RouterConfigRequest
from ...types import Response


def _get_kwargs(
    *,
    body: RouterConfigRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/router/configs",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | RouterConfig | None:
    if response.status_code == 201:
        response_201 = RouterConfig.from_dict(response.json())

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
) -> Response[ErrorResponse | RouterConfig]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: RouterConfigRequest,
) -> Response[ErrorResponse | RouterConfig]:
    """Store a named set of per-modality routing options

     A modality block that names no target falls back to what a session falls back to, so a config only
    has to say what it wants changed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | RouterConfig]
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
    body: RouterConfigRequest,
) -> ErrorResponse | RouterConfig | None:
    """Store a named set of per-modality routing options

     A modality block that names no target falls back to what a session falls back to, so a config only
    has to say what it wants changed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | RouterConfig
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: RouterConfigRequest,
) -> Response[ErrorResponse | RouterConfig]:
    """Store a named set of per-modality routing options

     A modality block that names no target falls back to what a session falls back to, so a config only
    has to say what it wants changed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | RouterConfig]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: RouterConfigRequest,
) -> ErrorResponse | RouterConfig | None:
    """Store a named set of per-modality routing options

     A modality block that names no target falls back to what a session falls back to, so a config only
    has to say what it wants changed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | RouterConfig
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
