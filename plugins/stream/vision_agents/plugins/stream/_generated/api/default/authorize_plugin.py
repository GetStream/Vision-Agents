from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.authorize_plugin_request import AuthorizePluginRequest
from ...models.authorize_plugin_response import AuthorizePluginResponse
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    plugin_id: str,
    *,
    body: AuthorizePluginRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/configs/{id}/plugins/{plugin_id}/authorize".format(
            id=quote(str(id), safe=""),
            plugin_id=quote(str(plugin_id), safe=""),
        ),
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AuthorizePluginResponse | Error | None:
    if response.status_code == 200:
        response_200 = AuthorizePluginResponse.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[AuthorizePluginResponse | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AuthorizePluginRequest | Unset = UNSET,
) -> Response[AuthorizePluginResponse | Error]:
    """Start a plugin login

     Discovers the MCP server's OAuth endpoints and returns the URL the browser should open. Shopify and
    Salesforce need an instance url, because they have no single global host.

    Args:
        id (str):
        plugin_id (str):
        body (AuthorizePluginRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AuthorizePluginResponse | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        plugin_id=plugin_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AuthorizePluginRequest | Unset = UNSET,
) -> AuthorizePluginResponse | Error | None:
    """Start a plugin login

     Discovers the MCP server's OAuth endpoints and returns the URL the browser should open. Shopify and
    Salesforce need an instance url, because they have no single global host.

    Args:
        id (str):
        plugin_id (str):
        body (AuthorizePluginRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AuthorizePluginResponse | Error
    """

    return sync_detailed(
        id=id,
        plugin_id=plugin_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    id: str,
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AuthorizePluginRequest | Unset = UNSET,
) -> Response[AuthorizePluginResponse | Error]:
    """Start a plugin login

     Discovers the MCP server's OAuth endpoints and returns the URL the browser should open. Shopify and
    Salesforce need an instance url, because they have no single global host.

    Args:
        id (str):
        plugin_id (str):
        body (AuthorizePluginRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AuthorizePluginResponse | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        plugin_id=plugin_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AuthorizePluginRequest | Unset = UNSET,
) -> AuthorizePluginResponse | Error | None:
    """Start a plugin login

     Discovers the MCP server's OAuth endpoints and returns the URL the browser should open. Shopify and
    Salesforce need an instance url, because they have no single global host.

    Args:
        id (str):
        plugin_id (str):
        body (AuthorizePluginRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AuthorizePluginResponse | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            plugin_id=plugin_id,
            client=client,
            body=body,
        )
    ).parsed
