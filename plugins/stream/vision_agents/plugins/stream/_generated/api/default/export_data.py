from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/data/export",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | str | None:
    if response.status_code == 200:
        response_200 = response.text
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
) -> Response[Error | str]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | str]:
    """Everything this app has, as newline-delimited JSON

     One line per row, `{"table": ..., "row": {...}}`, and a last line `{"cursor": ..., "customer": ...,
    "at": ...}`. The cursor comes last because it is also what says the export finished: a stream that
    broke halfway has no cursor line, so half a copy cannot be mistaken for a whole one. The rows are
    read at one moment rather than stitched together, and exporting starts recording changes so that
    `listDataChanges` carries on from exactly where this left off.
    Only the calling app's rows are here, and credentials are not: an API key secret, an OAuth access
    token and an OAuth refresh token stay with the deployment that holds them, so an imported plugin
    connection has to be authorized again. The audio behind a voice sample and a call recording lives in
    an object bucket rather than in this database; the rows naming those objects are here, and copying
    the bucket is yours to do.
    Server-side only, and refused entirely when the router runs with ROUTER_AUTH_MODE=noauth, where
    naming a customer is all it takes to be one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | str]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> Error | str | None:
    """Everything this app has, as newline-delimited JSON

     One line per row, `{"table": ..., "row": {...}}`, and a last line `{"cursor": ..., "customer": ...,
    "at": ...}`. The cursor comes last because it is also what says the export finished: a stream that
    broke halfway has no cursor line, so half a copy cannot be mistaken for a whole one. The rows are
    read at one moment rather than stitched together, and exporting starts recording changes so that
    `listDataChanges` carries on from exactly where this left off.
    Only the calling app's rows are here, and credentials are not: an API key secret, an OAuth access
    token and an OAuth refresh token stay with the deployment that holds them, so an imported plugin
    connection has to be authorized again. The audio behind a voice sample and a call recording lives in
    an object bucket rather than in this database; the rows naming those objects are here, and copying
    the bucket is yours to do.
    Server-side only, and refused entirely when the router runs with ROUTER_AUTH_MODE=noauth, where
    naming a customer is all it takes to be one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | str
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | str]:
    """Everything this app has, as newline-delimited JSON

     One line per row, `{"table": ..., "row": {...}}`, and a last line `{"cursor": ..., "customer": ...,
    "at": ...}`. The cursor comes last because it is also what says the export finished: a stream that
    broke halfway has no cursor line, so half a copy cannot be mistaken for a whole one. The rows are
    read at one moment rather than stitched together, and exporting starts recording changes so that
    `listDataChanges` carries on from exactly where this left off.
    Only the calling app's rows are here, and credentials are not: an API key secret, an OAuth access
    token and an OAuth refresh token stay with the deployment that holds them, so an imported plugin
    connection has to be authorized again. The audio behind a voice sample and a call recording lives in
    an object bucket rather than in this database; the rows naming those objects are here, and copying
    the bucket is yours to do.
    Server-side only, and refused entirely when the router runs with ROUTER_AUTH_MODE=noauth, where
    naming a customer is all it takes to be one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | str]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> Error | str | None:
    """Everything this app has, as newline-delimited JSON

     One line per row, `{"table": ..., "row": {...}}`, and a last line `{"cursor": ..., "customer": ...,
    "at": ...}`. The cursor comes last because it is also what says the export finished: a stream that
    broke halfway has no cursor line, so half a copy cannot be mistaken for a whole one. The rows are
    read at one moment rather than stitched together, and exporting starts recording changes so that
    `listDataChanges` carries on from exactly where this left off.
    Only the calling app's rows are here, and credentials are not: an API key secret, an OAuth access
    token and an OAuth refresh token stay with the deployment that holds them, so an imported plugin
    connection has to be authorized again. The audio behind a voice sample and a call recording lives in
    an object bucket rather than in this database; the rows naming those objects are here, and copying
    the bucket is yours to do.
    Server-side only, and refused entirely when the router runs with ROUTER_AUTH_MODE=noauth, where
    naming a customer is all it takes to be one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | str
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
