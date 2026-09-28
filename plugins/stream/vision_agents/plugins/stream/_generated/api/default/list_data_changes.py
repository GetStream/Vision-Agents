from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.data_change_page import DataChangePage
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    after: int | Unset = UNSET,
    limit: int | Unset = 500,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["after"] = after

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/data/changes",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> DataChangePage | Error | None:
    if response.status_code == 200:
        response_200 = DataChangePage.from_dict(response.json())

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

    if response.status_code == 410:
        response_410 = Error.from_dict(response.json())

        return response_410

    if response.status_code == 503:
        response_503 = Error.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[DataChangePage | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = UNSET,
    limit: int | Unset = 500,
) -> Response[DataChangePage | Error]:
    """What has happened to this app's rows since a cursor

     Oldest first, for replaying onto the deployment that took the export. A change is only returned once
    every transaction older than it has committed, so following the cursor never steps over a row, and a
    change carries the row as it now reads rather than the columns that changed, so applying one twice
    is the same as applying it once.
    Changes are only recorded for an app that has exported, and only for as long as the deployment's
    retention window. A cursor older than what is still kept is answered 410, which means export again.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        after (int | Unset):
        limit (int | Unset):  Default: 500.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DataChangePage | Error]
    """

    kwargs = _get_kwargs(
        after=after,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = UNSET,
    limit: int | Unset = 500,
) -> DataChangePage | Error | None:
    """What has happened to this app's rows since a cursor

     Oldest first, for replaying onto the deployment that took the export. A change is only returned once
    every transaction older than it has committed, so following the cursor never steps over a row, and a
    change carries the row as it now reads rather than the columns that changed, so applying one twice
    is the same as applying it once.
    Changes are only recorded for an app that has exported, and only for as long as the deployment's
    retention window. A cursor older than what is still kept is answered 410, which means export again.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        after (int | Unset):
        limit (int | Unset):  Default: 500.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DataChangePage | Error
    """

    return sync_detailed(
        client=client,
        after=after,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = UNSET,
    limit: int | Unset = 500,
) -> Response[DataChangePage | Error]:
    """What has happened to this app's rows since a cursor

     Oldest first, for replaying onto the deployment that took the export. A change is only returned once
    every transaction older than it has committed, so following the cursor never steps over a row, and a
    change carries the row as it now reads rather than the columns that changed, so applying one twice
    is the same as applying it once.
    Changes are only recorded for an app that has exported, and only for as long as the deployment's
    retention window. A cursor older than what is still kept is answered 410, which means export again.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        after (int | Unset):
        limit (int | Unset):  Default: 500.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DataChangePage | Error]
    """

    kwargs = _get_kwargs(
        after=after,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = UNSET,
    limit: int | Unset = 500,
) -> DataChangePage | Error | None:
    """What has happened to this app's rows since a cursor

     Oldest first, for replaying onto the deployment that took the export. A change is only returned once
    every transaction older than it has committed, so following the cursor never steps over a row, and a
    change carries the row as it now reads rather than the columns that changed, so applying one twice
    is the same as applying it once.
    Changes are only recorded for an app that has exported, and only for as long as the deployment's
    retention window. A cursor older than what is still kept is answered 410, which means export again.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        after (int | Unset):
        limit (int | Unset):  Default: 500.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DataChangePage | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            after=after,
            limit=limit,
        )
    ).parsed
