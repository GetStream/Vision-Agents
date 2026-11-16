from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.limit_grant import LimitGrant
from ...models.limit_grant_unrevocation import LimitGrantUnrevocation
from ...types import Response


def _get_kwargs(
    grant_id: int,
    *,
    body: LimitGrantUnrevocation,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/spend/limits/grants/{grant_id}/unrevoke".format(
            grant_id=quote(str(grant_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | LimitGrant | None:
    if response.status_code == 200:
        response_200 = LimitGrant.from_dict(response.json())

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
) -> Response[ErrorResponse | LimitGrant]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    grant_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: LimitGrantUnrevocation,
) -> Response[ErrorResponse | LimitGrant]:
    """Unrevoke a grant

     Makes a revoked grant live again: the next admission draws on it once a limit of its subject is
    spent, until it is used up, expires or is revoked again. The work admitted while it was revoked is
    not drawn from it. Unrevoking one that is not revoked changes nothing. An expired grant cannot be
    unrevoked (400 invalid_expiry), and an organization has at most 20 live grants (409
    too_many_grants), a used-up one not counted. The router that unrevoked it draws on it at once, the
    others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantUnrevocation): Why a revoked grant is live again.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitGrant]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    grant_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: LimitGrantUnrevocation,
) -> ErrorResponse | LimitGrant | None:
    """Unrevoke a grant

     Makes a revoked grant live again: the next admission draws on it once a limit of its subject is
    spent, until it is used up, expires or is revoked again. The work admitted while it was revoked is
    not drawn from it. Unrevoking one that is not revoked changes nothing. An expired grant cannot be
    unrevoked (400 invalid_expiry), and an organization has at most 20 live grants (409
    too_many_grants), a used-up one not counted. The router that unrevoked it draws on it at once, the
    others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantUnrevocation): Why a revoked grant is live again.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitGrant
    """

    return sync_detailed(
        grant_id=grant_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    grant_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: LimitGrantUnrevocation,
) -> Response[ErrorResponse | LimitGrant]:
    """Unrevoke a grant

     Makes a revoked grant live again: the next admission draws on it once a limit of its subject is
    spent, until it is used up, expires or is revoked again. The work admitted while it was revoked is
    not drawn from it. Unrevoking one that is not revoked changes nothing. An expired grant cannot be
    unrevoked (400 invalid_expiry), and an organization has at most 20 live grants (409
    too_many_grants), a used-up one not counted. The router that unrevoked it draws on it at once, the
    others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantUnrevocation): Why a revoked grant is live again.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitGrant]
    """

    kwargs = _get_kwargs(
        grant_id=grant_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    grant_id: int,
    *,
    client: AuthenticatedClient | Client,
    body: LimitGrantUnrevocation,
) -> ErrorResponse | LimitGrant | None:
    """Unrevoke a grant

     Makes a revoked grant live again: the next admission draws on it once a limit of its subject is
    spent, until it is used up, expires or is revoked again. The work admitted while it was revoked is
    not drawn from it. Unrevoking one that is not revoked changes nothing. An expired grant cannot be
    unrevoked (400 invalid_expiry), and an organization has at most 20 live grants (409
    too_many_grants), a used-up one not counted. The router that unrevoked it draws on it at once, the
    others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantUnrevocation): Why a revoked grant is live again.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitGrant
    """

    return (
        await asyncio_detailed(
            grant_id=grant_id,
            client=client,
            body=body,
        )
    ).parsed
