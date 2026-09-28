from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.authorize_connector_request import AuthorizeConnectorRequest
from ...models.connector_authorization import ConnectorAuthorization
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    body: AuthorizeConnectorRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connections/{id}/authorizations".format(
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
) -> ConnectorAuthorization | Error | None:
    if response.status_code == 200:
        response_200 = ConnectorAuthorization.from_dict(response.json())

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

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ConnectorAuthorization | Error]:
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
    body: AuthorizeConnectorRequest | Unset = UNSET,
) -> Response[ConnectorAuthorization | Error]:
    """Start provider consent for an account connection

     Starts an OAuth authorization attempt. Customer-created integrations such as Gong may use approved
    dynamic registration or submit both their client ID and secret; supplied credentials are write-only
    and stored encrypted for token exchange and refresh. Open the returned launch URL in a popup and
    deliver the handoff token to that popup with postMessage; the router then sets a short-lived,
    HttpOnly callback cookie on its own origin before redirecting the browser to the provider. Never put
    the handoff token in a URL.

    Args:
        id (str):
        body (AuthorizeConnectorRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorAuthorization | Error]
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
    body: AuthorizeConnectorRequest | Unset = UNSET,
) -> ConnectorAuthorization | Error | None:
    """Start provider consent for an account connection

     Starts an OAuth authorization attempt. Customer-created integrations such as Gong may use approved
    dynamic registration or submit both their client ID and secret; supplied credentials are write-only
    and stored encrypted for token exchange and refresh. Open the returned launch URL in a popup and
    deliver the handoff token to that popup with postMessage; the router then sets a short-lived,
    HttpOnly callback cookie on its own origin before redirecting the browser to the provider. Never put
    the handoff token in a URL.

    Args:
        id (str):
        body (AuthorizeConnectorRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorAuthorization | Error
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
    body: AuthorizeConnectorRequest | Unset = UNSET,
) -> Response[ConnectorAuthorization | Error]:
    """Start provider consent for an account connection

     Starts an OAuth authorization attempt. Customer-created integrations such as Gong may use approved
    dynamic registration or submit both their client ID and secret; supplied credentials are write-only
    and stored encrypted for token exchange and refresh. Open the returned launch URL in a popup and
    deliver the handoff token to that popup with postMessage; the router then sets a short-lived,
    HttpOnly callback cookie on its own origin before redirecting the browser to the provider. Never put
    the handoff token in a URL.

    Args:
        id (str):
        body (AuthorizeConnectorRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorAuthorization | Error]
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
    body: AuthorizeConnectorRequest | Unset = UNSET,
) -> ConnectorAuthorization | Error | None:
    """Start provider consent for an account connection

     Starts an OAuth authorization attempt. Customer-created integrations such as Gong may use approved
    dynamic registration or submit both their client ID and secret; supplied credentials are write-only
    and stored encrypted for token exchange and refresh. Open the returned launch URL in a popup and
    deliver the handoff token to that popup with postMessage; the router then sets a short-lived,
    HttpOnly callback cookie on its own origin before redirecting the browser to the provider. Never put
    the handoff token in a URL.

    Args:
        id (str):
        body (AuthorizeConnectorRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorAuthorization | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
