from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.limit_grant import LimitGrant
from ...models.limit_grant_input import LimitGrantInput
from ...types import Response


def _get_kwargs(
    *,
    body: LimitGrantInput,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/spend/limits/grants",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | LimitGrant | None:
    if response.status_code == 201:
        response_201 = LimitGrant.from_dict(response.json())

        return response_201

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
    *,
    client: AuthenticatedClient | Client,
    body: LimitGrantInput,
) -> Response[ErrorResponse | LimitGrant]:
    """Grant a subject a one-time amount past its limits

     Once one of the subject's own limits is spent, in any interval, its work is admitted on the grant
    until the grant is used up, expires or is revoked; work that fits under every limit draws nothing.
    It lifts the subject's own limits only: a parent's limit, credit and the operator's spend hard limit
    still refuse. The subject must have a limit (404 no_limit); expires_at must be in the future and at
    most a year away (400 invalid_expiry); an organization has at most 20 live grants (409
    too_many_grants). The router that made it draws on it at once, the others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        body (LimitGrantInput): A one-time grant to one subject, which names the subject's own
            ids, never a default's.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitGrant]
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
    body: LimitGrantInput,
) -> ErrorResponse | LimitGrant | None:
    """Grant a subject a one-time amount past its limits

     Once one of the subject's own limits is spent, in any interval, its work is admitted on the grant
    until the grant is used up, expires or is revoked; work that fits under every limit draws nothing.
    It lifts the subject's own limits only: a parent's limit, credit and the operator's spend hard limit
    still refuse. The subject must have a limit (404 no_limit); expires_at must be in the future and at
    most a year away (400 invalid_expiry); an organization has at most 20 live grants (409
    too_many_grants). The router that made it draws on it at once, the others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        body (LimitGrantInput): A one-time grant to one subject, which names the subject's own
            ids, never a default's.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitGrant
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: LimitGrantInput,
) -> Response[ErrorResponse | LimitGrant]:
    """Grant a subject a one-time amount past its limits

     Once one of the subject's own limits is spent, in any interval, its work is admitted on the grant
    until the grant is used up, expires or is revoked; work that fits under every limit draws nothing.
    It lifts the subject's own limits only: a parent's limit, credit and the operator's spend hard limit
    still refuse. The subject must have a limit (404 no_limit); expires_at must be in the future and at
    most a year away (400 invalid_expiry); an organization has at most 20 live grants (409
    too_many_grants). The router that made it draws on it at once, the others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        body (LimitGrantInput): A one-time grant to one subject, which names the subject's own
            ids, never a default's.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitGrant]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: LimitGrantInput,
) -> ErrorResponse | LimitGrant | None:
    """Grant a subject a one-time amount past its limits

     Once one of the subject's own limits is spent, in any interval, its work is admitted on the grant
    until the grant is used up, expires or is revoked; work that fits under every limit draws nothing.
    It lifts the subject's own limits only: a parent's limit, credit and the operator's spend hard limit
    still refuse. The subject must have a limit (404 no_limit); expires_at must be in the future and at
    most a year away (400 invalid_expiry); an organization has at most 20 live grants (409
    too_many_grants). The router that made it draws on it at once, the others within 5 s.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        body (LimitGrantInput): A one-time grant to one subject, which names the subject's own
            ids, never a default's.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitGrant
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
