from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.limit import Limit
from ...models.limit_change import LimitChange
from ...types import Response


def _get_kwargs(
    limit_id: int,
    *,
    body: LimitChange,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/v1/spend/limits/{limit_id}".format(
            limit_id=quote(str(limit_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | Limit | None:
    if response.status_code == 200:
        response_200 = Limit.from_dict(response.json())

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
) -> Response[ErrorResponse | Limit]:
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
    body: LimitChange,
) -> Response[ErrorResponse | Limit]:
    """Change a spend limit's amount or note

     Only the amount and the note change. A higher amount is held to the parents' as a new limit is (400
    limit_above_parent). The router that wrote it holds to it at once, every other within 5 s.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):
        body (LimitChange): What changes about a spend limit: its amount and its note. A field
            left out keeps what is stored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Limit]
    """

    kwargs = _get_kwargs(
        limit_id=limit_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    limit_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: LimitChange,
) -> ErrorResponse | Limit | None:
    """Change a spend limit's amount or note

     Only the amount and the note change. A higher amount is held to the parents' as a new limit is (400
    limit_above_parent). The router that wrote it holds to it at once, every other within 5 s.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):
        body (LimitChange): What changes about a spend limit: its amount and its note. A field
            left out keeps what is stored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Limit
    """

    return sync_detailed(
        limit_id=limit_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    limit_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: LimitChange,
) -> Response[ErrorResponse | Limit]:
    """Change a spend limit's amount or note

     Only the amount and the note change. A higher amount is held to the parents' as a new limit is (400
    limit_above_parent). The router that wrote it holds to it at once, every other within 5 s.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):
        body (LimitChange): What changes about a spend limit: its amount and its note. A field
            left out keeps what is stored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Limit]
    """

    kwargs = _get_kwargs(
        limit_id=limit_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    limit_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: LimitChange,
) -> ErrorResponse | Limit | None:
    """Change a spend limit's amount or note

     Only the amount and the note change. A higher amount is held to the parents' as a new limit is (400
    limit_above_parent). The router that wrote it holds to it at once, every other within 5 s.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        limit_id (int):
        body (LimitChange): What changes about a spend limit: its amount and its note. A field
            left out keeps what is stored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Limit
    """

    return (
        await asyncio_detailed(
            limit_id=limit_id,
            client=client,
            body=body,
        )
    ).parsed
