from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection_token import ConnectionToken
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connections/{id}/token".format(
            id=quote(str(id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectionToken | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ConnectionToken.from_dict(response.json())

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
) -> Response[ConnectionToken | ErrorResponse]:
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
) -> Response[ConnectionToken | ErrorResponse]:
    """Export a connection's access token

     The connection's current access credential, for the app's backend to call the provider with
    directly: an OAuth access token, renewed first when it is about to expire, or an API key. A refresh
    token is never exported. Only the customer's own provider app exports: an oauth2_code connection
    exports when its grant was issued to the client the app registered itself, and that client is still
    the connector's; a grant issued to Stream's app, or to one the router created, is refused with a
    403. An api_key connection always exports, since the key is the app's own. Other schemes are
    refused. Each export is recorded in the connector audit as token_export. Who may export it is who
    may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectionToken | ErrorResponse]
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
) -> ConnectionToken | ErrorResponse | None:
    """Export a connection's access token

     The connection's current access credential, for the app's backend to call the provider with
    directly: an OAuth access token, renewed first when it is about to expire, or an API key. A refresh
    token is never exported. Only the customer's own provider app exports: an oauth2_code connection
    exports when its grant was issued to the client the app registered itself, and that client is still
    the connector's; a grant issued to Stream's app, or to one the router created, is refused with a
    403. An api_key connection always exports, since the key is the app's own. Other schemes are
    refused. Each export is recorded in the connector audit as token_export. Who may export it is who
    may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectionToken | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ConnectionToken | ErrorResponse]:
    """Export a connection's access token

     The connection's current access credential, for the app's backend to call the provider with
    directly: an OAuth access token, renewed first when it is about to expire, or an API key. A refresh
    token is never exported. Only the customer's own provider app exports: an oauth2_code connection
    exports when its grant was issued to the client the app registered itself, and that client is still
    the connector's; a grant issued to Stream's app, or to one the router created, is refused with a
    403. An api_key connection always exports, since the key is the app's own. Other schemes are
    refused. Each export is recorded in the connector audit as token_export. Who may export it is who
    may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectionToken | ErrorResponse]
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
) -> ConnectionToken | ErrorResponse | None:
    """Export a connection's access token

     The connection's current access credential, for the app's backend to call the provider with
    directly: an OAuth access token, renewed first when it is about to expire, or an API key. A refresh
    token is never exported. Only the customer's own provider app exports: an oauth2_code connection
    exports when its grant was issued to the client the app registered itself, and that client is still
    the connector's; a grant issued to Stream's app, or to one the router created, is refused with a
    403. An api_key connection always exports, since the key is the app's own. Other schemes are
    refused. Each export is recorded in the connector audit as token_export. Who may export it is who
    may read it.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectionToken | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
