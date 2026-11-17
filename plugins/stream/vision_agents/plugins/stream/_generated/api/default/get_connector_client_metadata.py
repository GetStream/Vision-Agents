from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.get_connector_client_metadata_response_200 import (
    GetConnectorClientMetadataResponse200,
)
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/.well-known/oauth-client-metadata",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | GetConnectorClientMetadataResponse200 | None:
    if response.status_code == 200:
        response_200 = GetConnectorClientMetadataResponse200.from_dict(response.json())

        return response_200

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
) -> Response[ErrorResponse | GetConnectorClientMetadataResponse200]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[ErrorResponse | GetConnectorClientMetadataResponse200]:
    """The router's OAuth client metadata

     The OAuth Client ID Metadata Document a provider that supports it fetches, at the URL that is the
    router's client_id. Served only when ROUTER_PUBLIC_URL is https. Unauthenticated because the
    provider fetches it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | GetConnectorClientMetadataResponse200]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | GetConnectorClientMetadataResponse200 | None:
    """The router's OAuth client metadata

     The OAuth Client ID Metadata Document a provider that supports it fetches, at the URL that is the
    router's client_id. Served only when ROUTER_PUBLIC_URL is https. Unauthenticated because the
    provider fetches it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | GetConnectorClientMetadataResponse200
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[ErrorResponse | GetConnectorClientMetadataResponse200]:
    """The router's OAuth client metadata

     The OAuth Client ID Metadata Document a provider that supports it fetches, at the URL that is the
    router's client_id. Served only when ROUTER_PUBLIC_URL is https. Unauthenticated because the
    provider fetches it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | GetConnectorClientMetadataResponse200]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | GetConnectorClientMetadataResponse200 | None:
    """The router's OAuth client metadata

     The OAuth Client ID Metadata Document a provider that supports it fetches, at the URL that is the
    router's client_id. Served only when ROUTER_PUBLIC_URL is https. Unauthenticated because the
    provider fetches it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | GetConnectorClientMetadataResponse200
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
