from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.data_import import DataImport
from ...models.error import Error
from ...types import File, Response


def _get_kwargs(
    *,
    body: File,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/data/import",
    }

    _kwargs["content"] = body.payload
    headers["Content-Type"] = "application/octet-stream"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> DataImport | Error | None:
    if response.status_code == 200:
        response_200 = DataImport.from_dict(response.json())

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

    if response.status_code == 503:
        response_503 = Error.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[DataImport | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: File,
) -> Response[DataImport | Error]:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    Importing is idempotent: the same export applied twice leaves what applying it once would.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DataImport | Error]
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
    body: File,
) -> DataImport | Error | None:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    Importing is idempotent: the same export applied twice leaves what applying it once would.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DataImport | Error
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: File,
) -> Response[DataImport | Error]:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    Importing is idempotent: the same export applied twice leaves what applying it once would.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DataImport | Error]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: File,
) -> DataImport | Error | None:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    Importing is idempotent: the same export applied twice leaves what applying it once would.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DataImport | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
