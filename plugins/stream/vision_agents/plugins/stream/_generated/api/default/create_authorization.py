from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.authorization import Authorization
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connections/{id}/authorizations".format(
            id=quote(str(id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Authorization | ErrorResponse | None:
    if response.status_code == 201:
        response_201 = Authorization.from_dict(response.json())

        return response_201

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

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
) -> Response[Authorization | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Authorization | ErrorResponse]:
    """Start a consent

     Starts the provider's consent for a connection: a consent for a pending one, a reconnect for one
    connected before. Open launch_url in a popup from the dashboard and post it handoff_token when it
    says it is ready; the browser then goes to the provider and comes back to the router, which stores
    the grant and sends the browser to the dashboard with connection_id and status (connected, denied,
    failed or account_mismatch). A reconnect that comes back with another provider account keeps the old
    grant. Who may start it is who may read the connection. Needs ROUTER_PUBLIC_URL, where the provider
    sends the browser back to.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Authorization | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Authorization | ErrorResponse | None:
    """Start a consent

     Starts the provider's consent for a connection: a consent for a pending one, a reconnect for one
    connected before. Open launch_url in a popup from the dashboard and post it handoff_token when it
    says it is ready; the browser then goes to the provider and comes back to the router, which stores
    the grant and sends the browser to the dashboard with connection_id and status (connected, denied,
    failed or account_mismatch). A reconnect that comes back with another provider account keeps the old
    grant. Who may start it is who may read the connection. Needs ROUTER_PUBLIC_URL, where the provider
    sends the browser back to.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Authorization | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Authorization | ErrorResponse]:
    """Start a consent

     Starts the provider's consent for a connection: a consent for a pending one, a reconnect for one
    connected before. Open launch_url in a popup from the dashboard and post it handoff_token when it
    says it is ready; the browser then goes to the provider and comes back to the router, which stores
    the grant and sends the browser to the dashboard with connection_id and status (connected, denied,
    failed or account_mismatch). A reconnect that comes back with another provider account keeps the old
    grant. Who may start it is who may read the connection. Needs ROUTER_PUBLIC_URL, where the provider
    sends the browser back to.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Authorization | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Authorization | ErrorResponse | None:
    """Start a consent

     Starts the provider's consent for a connection: a consent for a pending one, a reconnect for one
    connected before. Open launch_url in a popup from the dashboard and post it handoff_token when it
    says it is ready; the browser then goes to the provider and comes back to the router, which stores
    the grant and sends the browser to the dashboard with connection_id and status (connected, denied,
    failed or account_mismatch). A reconnect that comes back with another provider account keeps the old
    grant. Who may start it is who may read the connection. Needs ROUTER_PUBLIC_URL, where the provider
    sends the browser back to.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Authorization | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
