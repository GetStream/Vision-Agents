from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.user_policy_page import UserPolicyPage
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/policies/users",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | UserPolicyPage | None:
    if response.status_code == 200:
        response_200 = UserPolicyPage.from_dict(response.json())

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
) -> Response[ErrorResponse | UserPolicyPage]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | UserPolicyPage]:
    """List the calling app's users with a policy of their own

     By user id, in byte order. A user with no policy of their own is held to the app's quota, the
    organization's, or the router's default, and is not listed.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        limit (int | Unset): Up to 100. Omitted is 100.
        cursor (str | Unset): The next_cursor of the previous page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | UserPolicyPage]
    """

    kwargs = _get_kwargs(
        limit=limit,
        cursor=cursor,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | UserPolicyPage | None:
    """List the calling app's users with a policy of their own

     By user id, in byte order. A user with no policy of their own is held to the app's quota, the
    organization's, or the router's default, and is not listed.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        limit (int | Unset): Up to 100. Omitted is 100.
        cursor (str | Unset): The next_cursor of the previous page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | UserPolicyPage
    """

    return sync_detailed(
        client=client,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | UserPolicyPage]:
    """List the calling app's users with a policy of their own

     By user id, in byte order. A user with no policy of their own is held to the app's quota, the
    organization's, or the router's default, and is not listed.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        limit (int | Unset): Up to 100. Omitted is 100.
        cursor (str | Unset): The next_cursor of the previous page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | UserPolicyPage]
    """

    kwargs = _get_kwargs(
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | UserPolicyPage | None:
    """List the calling app's users with a policy of their own

     By user id, in byte order. A user with no policy of their own is held to the app's quota, the
    organization's, or the router's default, and is not listed.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        limit (int | Unset): Up to 100. Omitted is 100.
        cursor (str | Unset): The next_cursor of the previous page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | UserPolicyPage
    """

    return (
        await asyncio_detailed(
            client=client,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
