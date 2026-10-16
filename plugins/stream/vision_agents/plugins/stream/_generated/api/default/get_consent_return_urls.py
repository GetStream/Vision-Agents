from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.consent_return_urls import ConsentReturnUrls
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/consent-return-urls",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConsentReturnUrls | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ConsentReturnUrls.from_dict(response.json())

        return response_200

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

    if response.status_code == 501:
        response_501 = ErrorResponse.from_dict(response.json())

        return response_501

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ConsentReturnUrls | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[ConsentReturnUrls | ErrorResponse]:
    """Read the pages that may finish the app's connector consents

     The app's consent return URLs as they were registered, an empty list when there are none.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConsentReturnUrls | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> ConsentReturnUrls | ErrorResponse | None:
    """Read the pages that may finish the app's connector consents

     The app's consent return URLs as they were registered, an empty list when there are none.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConsentReturnUrls | ErrorResponse
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[ConsentReturnUrls | ErrorResponse]:
    """Read the pages that may finish the app's connector consents

     The app's consent return URLs as they were registered, an empty list when there are none.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConsentReturnUrls | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> ConsentReturnUrls | ErrorResponse | None:
    """Read the pages that may finish the app's connector consents

     The app's consent return URLs as they were registered, an empty list when there are none.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConsentReturnUrls | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
