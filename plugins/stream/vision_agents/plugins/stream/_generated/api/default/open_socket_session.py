from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/socket",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 101:
        response_101 = cast(Any, None)
        return response_101

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
) -> Response[Any | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Hold a voice conversation over the socket itself, with no call

     A WebSocket, which OpenAPI cannot describe past the upgrade. Text frames are JSON objects carrying a
    `type`.
    The client's first frame is `start`, with `session` (a `CreateSessionRequest`) and an optional
    `sample_rate`, 16000 when left out. `call_id` may be left out: the router makes one up for the
    records. A `text` session is refused, because the socket carries audio. A field that `createSession`
    refuses from an end user's device is refused here too: `history` is server-side only.
    The server answers `session`, with the `Session` and the `sample_rate` in use. Then binary frames
    are PCM16 mono at that rate in both directions: the caller's audio in, and the agent's speech out at
    the pace it would be heard on a call. A `cleared` frame says speech already sent was thrown away
    because the caller cut in. Tool calls and every other event go over the session's events socket, as
    they do for a call.
    A refused start is an `error` frame with `error`, the message, and the socket closes. An `error`
    frame for a field refused from a device also carries `code` and `error_type`, the `code` and `type`
    that `createSession` answers the same field with.
    The session lasts as long as the socket. Closing the socket, or sending `stop`, ends the
    conversation. A conversation that ends closes the socket.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Hold a voice conversation over the socket itself, with no call

     A WebSocket, which OpenAPI cannot describe past the upgrade. Text frames are JSON objects carrying a
    `type`.
    The client's first frame is `start`, with `session` (a `CreateSessionRequest`) and an optional
    `sample_rate`, 16000 when left out. `call_id` may be left out: the router makes one up for the
    records. A `text` session is refused, because the socket carries audio. A field that `createSession`
    refuses from an end user's device is refused here too: `history` is server-side only.
    The server answers `session`, with the `Session` and the `sample_rate` in use. Then binary frames
    are PCM16 mono at that rate in both directions: the caller's audio in, and the agent's speech out at
    the pace it would be heard on a call. A `cleared` frame says speech already sent was thrown away
    because the caller cut in. Tool calls and every other event go over the session's events socket, as
    they do for a call.
    A refused start is an `error` frame with `error`, the message, and the socket closes. An `error`
    frame for a field refused from a device also carries `code` and `error_type`, the `code` and `type`
    that `createSession` answers the same field with.
    The session lasts as long as the socket. Closing the socket, or sending `stop`, ends the
    conversation. A conversation that ends closes the socket.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Hold a voice conversation over the socket itself, with no call

     A WebSocket, which OpenAPI cannot describe past the upgrade. Text frames are JSON objects carrying a
    `type`.
    The client's first frame is `start`, with `session` (a `CreateSessionRequest`) and an optional
    `sample_rate`, 16000 when left out. `call_id` may be left out: the router makes one up for the
    records. A `text` session is refused, because the socket carries audio. A field that `createSession`
    refuses from an end user's device is refused here too: `history` is server-side only.
    The server answers `session`, with the `Session` and the `sample_rate` in use. Then binary frames
    are PCM16 mono at that rate in both directions: the caller's audio in, and the agent's speech out at
    the pace it would be heard on a call. A `cleared` frame says speech already sent was thrown away
    because the caller cut in. Tool calls and every other event go over the session's events socket, as
    they do for a call.
    A refused start is an `error` frame with `error`, the message, and the socket closes. An `error`
    frame for a field refused from a device also carries `code` and `error_type`, the `code` and `type`
    that `createSession` answers the same field with.
    The session lasts as long as the socket. Closing the socket, or sending `stop`, ends the
    conversation. A conversation that ends closes the socket.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Hold a voice conversation over the socket itself, with no call

     A WebSocket, which OpenAPI cannot describe past the upgrade. Text frames are JSON objects carrying a
    `type`.
    The client's first frame is `start`, with `session` (a `CreateSessionRequest`) and an optional
    `sample_rate`, 16000 when left out. `call_id` may be left out: the router makes one up for the
    records. A `text` session is refused, because the socket carries audio. A field that `createSession`
    refuses from an end user's device is refused here too: `history` is server-side only.
    The server answers `session`, with the `Session` and the `sample_rate` in use. Then binary frames
    are PCM16 mono at that rate in both directions: the caller's audio in, and the agent's speech out at
    the pace it would be heard on a call. A `cleared` frame says speech already sent was thrown away
    because the caller cut in. Tool calls and every other event go over the session's events socket, as
    they do for a call.
    A refused start is an `error` frame with `error`, the message, and the socket closes. An `error`
    frame for a field refused from a device also carries `code` and `error_type`, the `code` and `type`
    that `createSession` answers the same field with.
    The session lasts as long as the socket. Closing the socket, or sending `stop`, ends the
    conversation. A conversation that ends closes the socket.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
