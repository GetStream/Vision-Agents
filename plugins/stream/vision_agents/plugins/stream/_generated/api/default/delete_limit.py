from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    limit_id: int,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/spend/limits/{limit_id}".format(
            limit_id=quote(str(limit_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

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

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    limit_id: int,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Delete a spend limit

     The subject is no longer held to it; a default at a level above takes over where there is one.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        limit_id=limit_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    limit_id: int,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Delete a spend limit

     The subject is no longer held to it; a default at a level above takes over where there is one.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        limit_id=limit_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    limit_id: int,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Delete a spend limit

     The subject is no longer held to it; a default at a level above takes over where there is one.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        limit_id=limit_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    limit_id: int,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Delete a spend limit

     The subject is no longer held to it; a default at a level above takes over where there is one.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            limit_id=limit_id,
            client=client,
        )
    ).parsed
