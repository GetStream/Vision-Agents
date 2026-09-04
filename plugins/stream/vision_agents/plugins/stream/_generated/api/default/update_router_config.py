from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.router_config import RouterConfig
from ...models.router_config_request import RouterConfigRequest
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: RouterConfigRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/router/configs/{id}".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | RouterConfig | None:
    if response.status_code == 200:
        response_200 = RouterConfig.from_dict(response.json())

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

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | RouterConfig]:
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
    body: RouterConfigRequest,
) -> Response[Error | RouterConfig]:
    """Replace a router config

     Every field is written, so the body is what the config now is rather than what changed about it.
    Sockets already open keep the options they were started with.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | RouterConfig]
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
    body: RouterConfigRequest,
) -> Error | RouterConfig | None:
    """Replace a router config

     Every field is written, so the body is what the config now is rather than what changed about it.
    Sockets already open keep the options they were started with.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | RouterConfig
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
    body: RouterConfigRequest,
) -> Response[Error | RouterConfig]:
    """Replace a router config

     Every field is written, so the body is what the config now is rather than what changed about it.
    Sockets already open keep the options they were started with.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | RouterConfig]
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
    body: RouterConfigRequest,
) -> Error | RouterConfig | None:
    """Replace a router config

     Every field is written, so the body is what the config now is rather than what changed about it.
    Sockets already open keep the options they were started with.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (RouterConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | RouterConfig
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
