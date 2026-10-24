from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.limit_grant import LimitGrant
from ...models.limit_grant_revocation import LimitGrantRevocation
from ...types import Response


def _get_kwargs(
    grant_id: int,
    *,
    body: LimitGrantRevocation,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/spend/limits/grants/{grant_id}/revoke".format(
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
    body: LimitGrantRevocation,
) -> Response[ErrorResponse | LimitGrant]:
    """Revoke a grant

     The next admission no longer draws on it; work already admitted finishes and still counts. The grant
    and what it paid for stay. Revoking it again changes nothing.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantRevocation): Why a grant ends now.

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
    body: LimitGrantRevocation,
) -> ErrorResponse | LimitGrant | None:
    """Revoke a grant

     The next admission no longer draws on it; work already admitted finishes and still counts. The grant
    and what it paid for stay. Revoking it again changes nothing.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantRevocation): Why a grant ends now.

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
    body: LimitGrantRevocation,
) -> Response[ErrorResponse | LimitGrant]:
    """Revoke a grant

     The next admission no longer draws on it; work already admitted finishes and still counts. The grant
    and what it paid for stay. Revoking it again changes nothing.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantRevocation): Why a grant ends now.

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
    body: LimitGrantRevocation,
) -> ErrorResponse | LimitGrant | None:
    """Revoke a grant

     The next admission no longer draws on it; work already admitted finishes and still counts. The grant
    and what it paid for stay. Revoking it again changes nothing.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        grant_id (int):
        body (LimitGrantRevocation): Why a grant ends now.

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
