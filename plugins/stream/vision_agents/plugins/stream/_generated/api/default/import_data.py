from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.data_import import DataImport
from ...models.error_response import ErrorResponse
from ...types import UNSET, File, Response, Unset


def _get_kwargs(
    *,
    body: File,
    full: bool | Unset = False,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["full"] = full

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/data/import",
        "params": params,
    }

    _kwargs["content"] = body.payload
    headers["Content-Type"] = "application/octet-stream"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> DataImport | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = DataImport.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

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
) -> Response[DataImport | ErrorResponse]:
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
    full: bool | Unset = False,
) -> Response[DataImport | ErrorResponse]:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    An export's rows are written as the source had them, so an export is refused with 409 `conflict`,
    before any of its rows is written, for an app that already has a row here in a table of the shared
    database: it would undo what was done here since. `full=true` writes it over them anyway, and
    importing the same export twice that way leaves what importing it once would. Changes are applied
    whatever the app holds. An import that stopped partway is run again with `full=true`, since the app
    then holds its first rows.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        full (bool | Unset):  Default: False.
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DataImport | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
        full=full,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: File,
    full: bool | Unset = False,
) -> DataImport | ErrorResponse | None:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    An export's rows are written as the source had them, so an export is refused with 409 `conflict`,
    before any of its rows is written, for an app that already has a row here in a table of the shared
    database: it would undo what was done here since. `full=true` writes it over them anyway, and
    importing the same export twice that way leaves what importing it once would. Changes are applied
    whatever the app holds. An import that stopped partway is run again with `full=true`, since the app
    then holds its first rows.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        full (bool | Unset):  Default: False.
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DataImport | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
        full=full,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: File,
    full: bool | Unset = False,
) -> Response[DataImport | ErrorResponse]:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    An export's rows are written as the source had them, so an export is refused with 409 `conflict`,
    before any of its rows is written, for an app that already has a row here in a table of the shared
    database: it would undo what was done here since. `full=true` writes it over them anyway, and
    importing the same export twice that way leaves what importing it once would. Changes are applied
    whatever the app holds. An import that stopped partway is run again with `full=true`, since the app
    then holds its first rows.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        full (bool | Unset):  Default: False.
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DataImport | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
        full=full,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: File,
    full: bool | Unset = False,
) -> DataImport | ErrorResponse | None:
    """Write an export, or a batch of changes, into this deployment

     Takes what `exportData` produced, and the same lines with a `change` in place of a `row` for what
    `listDataChanges` returned. Every row is written under the calling app whatever the file says, so an
    export from one app cannot be imported into another's rows, and rows belonging to a customer through
    a parent are only written where that parent is the caller's.
    An export's rows are written as the source had them, so an export is refused with 409 `conflict`,
    before any of its rows is written, for an app that already has a row here in a table of the shared
    database: it would undo what was done here since. `full=true` writes it over them anyway, and
    importing the same export twice that way leaves what importing it once would. Changes are applied
    whatever the app holds. An import that stopped partway is run again with `full=true`, since the app
    then holds its first rows.
    Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.

    Args:
        full (bool | Unset):  Default: False.
        body (File):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DataImport | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            full=full,
        )
    ).parsed
