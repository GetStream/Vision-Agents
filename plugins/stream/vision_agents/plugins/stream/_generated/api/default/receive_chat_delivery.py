from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/chat/internal/messages",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = cast(Any, None)
        return response_200

    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400

    if response.status_code == 401:
        response_401 = cast(Any, None)
        return response_401

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if response.status_code == 413:
        response_413 = cast(Any, None)
        return response_413

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = cast(Any, None)
        return response_503

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
    """Receive a message Stream Chat delivers

     Where Stream Chat delivers each message written in an agent channel, which is answered as the
    message webhook answers it. The body is the message.new event with the app_id it was written in.
    Unauthenticated because Stream is not a customer: each delivery is signed as Standard Webhooks signs
    with a public key (webhook-id, webhook-timestamp within 5 minutes, webhook-signature v1a, ed25519
    over id.timestamp.body) with a key only Stream holds, so no app's API secret, token or key is
    enough. A deployment with no delivery key answers 404.

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
    """Receive a message Stream Chat delivers

     Where Stream Chat delivers each message written in an agent channel, which is answered as the
    message webhook answers it. The body is the message.new event with the app_id it was written in.
    Unauthenticated because Stream is not a customer: each delivery is signed as Standard Webhooks signs
    with a public key (webhook-id, webhook-timestamp within 5 minutes, webhook-signature v1a, ed25519
    over id.timestamp.body) with a key only Stream holds, so no app's API secret, token or key is
    enough. A deployment with no delivery key answers 404.

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
    """Receive a message Stream Chat delivers

     Where Stream Chat delivers each message written in an agent channel, which is answered as the
    message webhook answers it. The body is the message.new event with the app_id it was written in.
    Unauthenticated because Stream is not a customer: each delivery is signed as Standard Webhooks signs
    with a public key (webhook-id, webhook-timestamp within 5 minutes, webhook-signature v1a, ed25519
    over id.timestamp.body) with a key only Stream holds, so no app's API secret, token or key is
    enough. A deployment with no delivery key answers 404.

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
    """Receive a message Stream Chat delivers

     Where Stream Chat delivers each message written in an agent channel, which is answered as the
    message webhook answers it. The body is the message.new event with the app_id it was written in.
    Unauthenticated because Stream is not a customer: each delivery is signed as Standard Webhooks signs
    with a public key (webhook-id, webhook-timestamp within 5 minutes, webhook-signature v1a, ed25519
    over id.timestamp.body) with a key only Stream holds, so no app's API secret, token or key is
    enough. A deployment with no delivery key answers 404.

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
