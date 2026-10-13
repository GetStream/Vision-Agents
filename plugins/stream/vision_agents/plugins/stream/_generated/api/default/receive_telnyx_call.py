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
        "url": "/v1/phone/hooks/telnyx",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
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
    """Receive a Telnyx call

     Where Telnyx asks what to do with a call to a number on this router's TeXML application.
    Unauthenticated because Telnyx is not a customer: each call is checked against the account's Ed25519
    signature, and the answer dials that number's own trunk and no other. A deployment with no Telnyx
    key answers 404, so the path never hands out trunk credentials unsigned.

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
    """Receive a Telnyx call

     Where Telnyx asks what to do with a call to a number on this router's TeXML application.
    Unauthenticated because Telnyx is not a customer: each call is checked against the account's Ed25519
    signature, and the answer dials that number's own trunk and no other. A deployment with no Telnyx
    key answers 404, so the path never hands out trunk credentials unsigned.

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
    """Receive a Telnyx call

     Where Telnyx asks what to do with a call to a number on this router's TeXML application.
    Unauthenticated because Telnyx is not a customer: each call is checked against the account's Ed25519
    signature, and the answer dials that number's own trunk and no other. A deployment with no Telnyx
    key answers 404, so the path never hands out trunk credentials unsigned.

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
    """Receive a Telnyx call

     Where Telnyx asks what to do with a call to a number on this router's TeXML application.
    Unauthenticated because Telnyx is not a customer: each call is checked against the account's Ed25519
    signature, and the answer dials that number's own trunk and no other. A deployment with no Telnyx
    key answers 404, so the path never hands out trunk credentials unsigned.

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
