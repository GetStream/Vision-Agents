from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.consent_return_urls import ConsentReturnUrls
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    *,
    body: ConsentReturnUrls,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/consent-return-urls",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
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
    body: ConsentReturnUrls,
) -> Response[ConsentReturnUrls | ErrorResponse]:
    """Set the pages that may finish the app's connector consents

     Replaces the app's whole list of consent return URLs with the one sent, and answers it. An empty
    list removes them all. Other app settings are left as they are.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConsentReturnUrls): The pages of the app's own that a connector consent may return
            to, besides the dashboard. Nothing reads the list yet: every consent still finishes on the
            dashboard.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConsentReturnUrls | ErrorResponse]
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
    body: ConsentReturnUrls,
) -> ConsentReturnUrls | ErrorResponse | None:
    """Set the pages that may finish the app's connector consents

     Replaces the app's whole list of consent return URLs with the one sent, and answers it. An empty
    list removes them all. Other app settings are left as they are.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConsentReturnUrls): The pages of the app's own that a connector consent may return
            to, besides the dashboard. Nothing reads the list yet: every consent still finishes on the
            dashboard.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConsentReturnUrls | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConsentReturnUrls,
) -> Response[ConsentReturnUrls | ErrorResponse]:
    """Set the pages that may finish the app's connector consents

     Replaces the app's whole list of consent return URLs with the one sent, and answers it. An empty
    list removes them all. Other app settings are left as they are.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConsentReturnUrls): The pages of the app's own that a connector consent may return
            to, besides the dashboard. Nothing reads the list yet: every consent still finishes on the
            dashboard.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConsentReturnUrls | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ConsentReturnUrls,
) -> ConsentReturnUrls | ErrorResponse | None:
    """Set the pages that may finish the app's connector consents

     Replaces the app's whole list of consent return URLs with the one sent, and answers it. An empty
    list removes them all. Other app settings are left as they are.

    Answered 501 by a router that does not run with ROUTER_AUTH_MODE=api_key, which keeps no settings
    for the app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ConsentReturnUrls): The pages of the app's own that a connector consent may return
            to, besides the dashboard. Nothing reads the list yet: every consent still finishes on the
            dashboard.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConsentReturnUrls | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
