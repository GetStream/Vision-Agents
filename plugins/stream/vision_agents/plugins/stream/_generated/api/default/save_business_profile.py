from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.business_profile import BusinessProfile
from ...models.business_profile_request import BusinessProfileRequest
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: BusinessProfileRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/phone/business-profile",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> BusinessProfile | Error | None:
    if response.status_code == 200:
        response_200 = BusinessProfile.from_dict(response.json())

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

    if response.status_code == 500:
        response_500 = Error.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[BusinessProfile | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: BusinessProfileRequest | Unset = UNSET,
) -> Response[BusinessProfile | Error]:
    """Save the business profile

     Replaces who the app says it is. Every use case is registered under it, so submitting one checks it
    is complete.

    Server-side only.

    Args:
        body (BusinessProfileRequest | Unset): Who the app is: written once and reused by every
            channel it registers for. Nothing is required to save it; submitting a use case says what
            is missing.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BusinessProfile | Error]
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
    body: BusinessProfileRequest | Unset = UNSET,
) -> BusinessProfile | Error | None:
    """Save the business profile

     Replaces who the app says it is. Every use case is registered under it, so submitting one checks it
    is complete.

    Server-side only.

    Args:
        body (BusinessProfileRequest | Unset): Who the app is: written once and reused by every
            channel it registers for. Nothing is required to save it; submitting a use case says what
            is missing.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BusinessProfile | Error
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: BusinessProfileRequest | Unset = UNSET,
) -> Response[BusinessProfile | Error]:
    """Save the business profile

     Replaces who the app says it is. Every use case is registered under it, so submitting one checks it
    is complete.

    Server-side only.

    Args:
        body (BusinessProfileRequest | Unset): Who the app is: written once and reused by every
            channel it registers for. Nothing is required to save it; submitting a use case says what
            is missing.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BusinessProfile | Error]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: BusinessProfileRequest | Unset = UNSET,
) -> BusinessProfile | Error | None:
    """Save the business profile

     Replaces who the app says it is. Every use case is registered under it, so submitting one checks it
    is complete.

    Server-side only.

    Args:
        body (BusinessProfileRequest | Unset): Who the app is: written once and reused by every
            channel it registers for. Nothing is required to save it; submitting a use case says what
            is missing.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BusinessProfile | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
