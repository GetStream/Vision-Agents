from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection_validation import ConnectionValidation
from ...models.connection_validation_request import ConnectionValidationRequest
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    body: ConnectionValidationRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connections/{id}/validate".format(
            id=quote(str(id), safe=""),
        ),
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectionValidation | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ConnectionValidation.from_dict(response.json())

        return response_200

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
) -> Response[ConnectionValidation | ErrorResponse]:
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
    body: ConnectionValidationRequest | Unset = UNSET,
) -> Response[ConnectionValidation | ErrorResponse]:
    """Validate a connection

     Gets the connection's credential, renewing it when it must, and asks the provider for its tools,
    which GET .../tools then shows. A connection that needs a reconnect says so without the provider
    being asked. The granted scopes are then checked against what the tools need (all of them, or those
    the body names): a grant that lacks some is needs_scopes with code connector_scope_required and the
    missing scopes. Who may validate it is who may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionValidationRequest | Unset): What a validate checks the grant's scopes
            against. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectionValidation | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionValidationRequest | Unset = UNSET,
) -> ConnectionValidation | ErrorResponse | None:
    """Validate a connection

     Gets the connection's credential, renewing it when it must, and asks the provider for its tools,
    which GET .../tools then shows. A connection that needs a reconnect says so without the provider
    being asked. The granted scopes are then checked against what the tools need (all of them, or those
    the body names): a grant that lacks some is needs_scopes with code connector_scope_required and the
    missing scopes. Who may validate it is who may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionValidationRequest | Unset): What a validate checks the grant's scopes
            against. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectionValidation | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionValidationRequest | Unset = UNSET,
) -> Response[ConnectionValidation | ErrorResponse]:
    """Validate a connection

     Gets the connection's credential, renewing it when it must, and asks the provider for its tools,
    which GET .../tools then shows. A connection that needs a reconnect says so without the provider
    being asked. The granted scopes are then checked against what the tools need (all of them, or those
    the body names): a grant that lacks some is needs_scopes with code connector_scope_required and the
    missing scopes. Who may validate it is who may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionValidationRequest | Unset): What a validate checks the grant's scopes
            against. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectionValidation | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionValidationRequest | Unset = UNSET,
) -> ConnectionValidation | ErrorResponse | None:
    """Validate a connection

     Gets the connection's credential, renewing it when it must, and asks the provider for its tools,
    which GET .../tools then shows. A connection that needs a reconnect says so without the provider
    being asked. The granted scopes are then checked against what the tools need (all of them, or those
    the body names): a grant that lacks some is needs_scopes with code connector_scope_required and the
    missing scopes. Who may validate it is who may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionValidationRequest | Unset): What a validate checks the grant's scopes
            against. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectionValidation | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
