from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.account import Account
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/account",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Account | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = Account.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Account | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Account | ErrorResponse]:
    """Read the organization's account

     The caller's organization's balance, credit, spend, status, tier, the threshold it is warned at and
    where that comes from, what Stream allows it, and Stream's spend hard limit on it with this window's
    spend. The balance is what admission counts, less the work in flight, so it agrees with X-Stream-
    Balance-Micros. With no usage service, or for an app in no organization, it answers 404
    billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Account | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> Account | ErrorResponse | None:
    """Read the organization's account

     The caller's organization's balance, credit, spend, status, tier, the threshold it is warned at and
    where that comes from, what Stream allows it, and Stream's spend hard limit on it with this window's
    spend. The balance is what admission counts, less the work in flight, so it agrees with X-Stream-
    Balance-Micros. With no usage service, or for an app in no organization, it answers 404
    billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Account | ErrorResponse
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Account | ErrorResponse]:
    """Read the organization's account

     The caller's organization's balance, credit, spend, status, tier, the threshold it is warned at and
    where that comes from, what Stream allows it, and Stream's spend hard limit on it with this window's
    spend. The balance is what admission counts, less the work in flight, so it agrees with X-Stream-
    Balance-Micros. With no usage service, or for an app in no organization, it answers 404
    billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Account | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> Account | ErrorResponse | None:
    """Read the organization's account

     The caller's organization's balance, credit, spend, status, tier, the threshold it is warned at and
    where that comes from, what Stream allows it, and Stream's spend hard limit on it with this window's
    spend. The balance is what admission counts, less the work in flight, so it agrees with X-Stream-
    Balance-Micros. With no usage service, or for an app in no organization, it answers 404
    billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Account | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
