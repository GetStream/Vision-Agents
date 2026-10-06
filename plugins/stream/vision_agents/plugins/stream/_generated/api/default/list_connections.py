from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection_owner_type import ConnectionOwnerType
from ...models.connection_page import ConnectionPage
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    owner_type: ConnectionOwnerType,
    connector_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_owner_type = owner_type.value
    params["owner_type"] = json_owner_type

    params["connector_id"] = connector_id

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/connections",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectionPage | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ConnectionPage.from_dict(response.json())

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

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ConnectionPage | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    owner_type: ConnectionOwnerType,
    connector_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ConnectionPage | ErrorResponse]:
    """List connections

     One owner's connections, newest first: the app's own, or those of the user the backend acts for.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        owner_type (ConnectionOwnerType): app is the app's own account, user one user's.
        connector_id (str | Unset): Keeps one connector's.
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectionPage | ErrorResponse]
    """

    kwargs = _get_kwargs(
        owner_type=owner_type,
        connector_id=connector_id,
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
    owner_type: ConnectionOwnerType,
    connector_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ConnectionPage | ErrorResponse | None:
    """List connections

     One owner's connections, newest first: the app's own, or those of the user the backend acts for.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        owner_type (ConnectionOwnerType): app is the app's own account, user one user's.
        connector_id (str | Unset): Keeps one connector's.
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectionPage | ErrorResponse
    """

    return sync_detailed(
        client=client,
        owner_type=owner_type,
        connector_id=connector_id,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    owner_type: ConnectionOwnerType,
    connector_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ConnectionPage | ErrorResponse]:
    """List connections

     One owner's connections, newest first: the app's own, or those of the user the backend acts for.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        owner_type (ConnectionOwnerType): app is the app's own account, user one user's.
        connector_id (str | Unset): Keeps one connector's.
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectionPage | ErrorResponse]
    """

    kwargs = _get_kwargs(
        owner_type=owner_type,
        connector_id=connector_id,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    owner_type: ConnectionOwnerType,
    connector_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ConnectionPage | ErrorResponse | None:
    """List connections

     One owner's connections, newest first: the app's own, or those of the user the backend acts for.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        owner_type (ConnectionOwnerType): app is the app's own account, user one user's.
        connector_id (str | Unset): Keeps one connector's.
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectionPage | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            owner_type=owner_type,
            connector_id=connector_id,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
