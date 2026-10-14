from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.plugin_client import PluginClient
from ...models.set_plugin_client_request import SetPluginClientRequest
from ...types import Response


def _get_kwargs(
    id: str,
    plugin_id: str,
    *,
    body: SetPluginClientRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/configs/{id}/plugins/{plugin_id}/client".format(
            id=quote(str(id), safe=""),
            plugin_id=quote(str(plugin_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | PluginClient | None:
    if response.status_code == 200:
        response_200 = PluginClient.from_dict(response.json())

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
) -> Response[ErrorResponse | PluginClient]:
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
    body: SetPluginClientRequest,
) -> Response[ErrorResponse | PluginClient]:
    """Set the OAuth client an agent logs a plugin in with

     Deprecated: use setConnectorOAuthClient, which sets the app's own client once per connector. The
    OAuth app the app registered with the provider, such as a Google Cloud client, used for this
    config's logins to the plugin: the app's own and every end user's. A plugin with client_required has
    no other way in. The secret is sealed and never returned. Replaces the client set before; a login
    made with that one keeps working until it has to be renewed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        plugin_id (str): A built-in catalog id such as google_calendar.
        body (SetPluginClientRequest): The OAuth client an app registered with a plugin's
            provider, with the redirect URI <public url>/v1/agents/plugins/callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | PluginClient]
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
    body: SetPluginClientRequest,
) -> ErrorResponse | PluginClient | None:
    """Set the OAuth client an agent logs a plugin in with

     Deprecated: use setConnectorOAuthClient, which sets the app's own client once per connector. The
    OAuth app the app registered with the provider, such as a Google Cloud client, used for this
    config's logins to the plugin: the app's own and every end user's. A plugin with client_required has
    no other way in. The secret is sealed and never returned. Replaces the client set before; a login
    made with that one keeps working until it has to be renewed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        plugin_id (str): A built-in catalog id such as google_calendar.
        body (SetPluginClientRequest): The OAuth client an app registered with a plugin's
            provider, with the redirect URI <public url>/v1/agents/plugins/callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | PluginClient
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
    body: SetPluginClientRequest,
) -> Response[ErrorResponse | PluginClient]:
    """Set the OAuth client an agent logs a plugin in with

     Deprecated: use setConnectorOAuthClient, which sets the app's own client once per connector. The
    OAuth app the app registered with the provider, such as a Google Cloud client, used for this
    config's logins to the plugin: the app's own and every end user's. A plugin with client_required has
    no other way in. The secret is sealed and never returned. Replaces the client set before; a login
    made with that one keeps working until it has to be renewed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        plugin_id (str): A built-in catalog id such as google_calendar.
        body (SetPluginClientRequest): The OAuth client an app registered with a plugin's
            provider, with the redirect URI <public url>/v1/agents/plugins/callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | PluginClient]
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
    body: SetPluginClientRequest,
) -> ErrorResponse | PluginClient | None:
    """Set the OAuth client an agent logs a plugin in with

     Deprecated: use setConnectorOAuthClient, which sets the app's own client once per connector. The
    OAuth app the app registered with the provider, such as a Google Cloud client, used for this
    config's logins to the plugin: the app's own and every end user's. A plugin with client_required has
    no other way in. The secret is sealed and never returned. Replaces the client set before; a login
    made with that one keeps working until it has to be renewed.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        plugin_id (str): A built-in catalog id such as google_calendar.
        body (SetPluginClientRequest): The OAuth client an app registered with a plugin's
            provider, with the redirect URI <public url>/v1/agents/plugins/callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | PluginClient
    """

    return (
        await asyncio_detailed(
            id=id,
            plugin_id=plugin_id,
            client=client,
            body=body,
        )
    ).parsed
