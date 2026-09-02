from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    capacity: int | Unset = 4,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["capacity"] = capacity

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/dispatch",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | Error | None:
    if response.status_code == 101:
        response_101 = cast(Any, None)
        return response_101

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
) -> Response[Any | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    capacity: int | Unset = 4,
) -> Response[Any | Error]:
    """Wait for inbound calls to answer, as a worker

     A WebSocket, which OpenAPI cannot describe past the upgrade. The worker connects here and waits,
    rather than being called, because the agent runs in the customer's own process and this service
    cannot reach into it.
    The socket opens with a `ready` frame naming the worker, and a `call` frame arrives for each call
    handed to it. The worker sends `load` so the pool can rank it, `accepted` or `rejected` per call,
    and `ping` to time the round trip itself.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    This is the clearest case of why: a worker is offered other people's callers, so anything that can
    open this socket can answer for the whole app. The auth type has no query parameter, so a browser
    cannot open one at all.

    Args:
        capacity (int | Unset):  Default: 4.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error]
    """

    kwargs = _get_kwargs(
        capacity=capacity,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    capacity: int | Unset = 4,
) -> Any | Error | None:
    """Wait for inbound calls to answer, as a worker

     A WebSocket, which OpenAPI cannot describe past the upgrade. The worker connects here and waits,
    rather than being called, because the agent runs in the customer's own process and this service
    cannot reach into it.
    The socket opens with a `ready` frame naming the worker, and a `call` frame arrives for each call
    handed to it. The worker sends `load` so the pool can rank it, `accepted` or `rejected` per call,
    and `ping` to time the round trip itself.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    This is the clearest case of why: a worker is offered other people's callers, so anything that can
    open this socket can answer for the whole app. The auth type has no query parameter, so a browser
    cannot open one at all.

    Args:
        capacity (int | Unset):  Default: 4.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error
    """

    return sync_detailed(
        client=client,
        capacity=capacity,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    capacity: int | Unset = 4,
) -> Response[Any | Error]:
    """Wait for inbound calls to answer, as a worker

     A WebSocket, which OpenAPI cannot describe past the upgrade. The worker connects here and waits,
    rather than being called, because the agent runs in the customer's own process and this service
    cannot reach into it.
    The socket opens with a `ready` frame naming the worker, and a `call` frame arrives for each call
    handed to it. The worker sends `load` so the pool can rank it, `accepted` or `rejected` per call,
    and `ping` to time the round trip itself.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    This is the clearest case of why: a worker is offered other people's callers, so anything that can
    open this socket can answer for the whole app. The auth type has no query parameter, so a browser
    cannot open one at all.

    Args:
        capacity (int | Unset):  Default: 4.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error]
    """

    kwargs = _get_kwargs(
        capacity=capacity,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    capacity: int | Unset = 4,
) -> Any | Error | None:
    """Wait for inbound calls to answer, as a worker

     A WebSocket, which OpenAPI cannot describe past the upgrade. The worker connects here and waits,
    rather than being called, because the agent runs in the customer's own process and this service
    cannot reach into it.
    The socket opens with a `ready` frame naming the worker, and a `call` frame arrives for each call
    handed to it. The worker sends `load` so the pool can rank it, `accepted` or `rejected` per call,
    and `ping` to time the round trip itself.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    This is the clearest case of why: a worker is offered other people's callers, so anything that can
    open this socket can answer for the whole app. The auth type has no query parameter, so a browser
    cannot open one at all.

    Args:
        capacity (int | Unset):  Default: 4.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            capacity=capacity,
        )
    ).parsed
