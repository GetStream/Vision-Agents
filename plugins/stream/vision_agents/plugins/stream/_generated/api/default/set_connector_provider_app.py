from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_provider_app import ConnectorProviderApp
from ...models.connector_provider_app_request import ConnectorProviderAppRequest
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: ConnectorProviderAppRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/connectors/{id}/provider-app".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectorProviderApp | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ConnectorProviderApp.from_dict(response.json())

        return response_200

    if response.status_code == 201:
        response_201 = ConnectorProviderApp.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 429:
        response_429 = ErrorResponse.from_dict(response.json())

        return response_429

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
) -> Response[ConnectorProviderApp | ErrorResponse]:
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
    body: ConnectorProviderAppRequest,
) -> Response[ConnectorProviderApp | ErrorResponse]:
    """Create or update the app the router keeps at the connector's provider

     Creates the customer's own Slack app in its workspace with Slack's apps.manifest.create, from the
    connector's scopes and events, the name given and this router's callback and events URLs, with token
    rotation on. The app's client is what every later consent of the connector's connections uses. It
    needs an app configuration token's refresh token the first time, which a workspace admin generates
    in Slack's app settings; the router rotates it before it expires and keeps it sealed. Putting it
    again changes nothing at Slack but the app's manifest: there is one app per customer and connector,
    never a second. A connector that does not authorize at Slack, or whose client.registration does not
    list managed, refuses it. No response carries a token or a secret.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack.
        body (ConnectorProviderAppRequest): The app the router creates and keeps in the customer's
            workspace. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorProviderApp | ErrorResponse]
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
    body: ConnectorProviderAppRequest,
) -> ConnectorProviderApp | ErrorResponse | None:
    """Create or update the app the router keeps at the connector's provider

     Creates the customer's own Slack app in its workspace with Slack's apps.manifest.create, from the
    connector's scopes and events, the name given and this router's callback and events URLs, with token
    rotation on. The app's client is what every later consent of the connector's connections uses. It
    needs an app configuration token's refresh token the first time, which a workspace admin generates
    in Slack's app settings; the router rotates it before it expires and keeps it sealed. Putting it
    again changes nothing at Slack but the app's manifest: there is one app per customer and connector,
    never a second. A connector that does not authorize at Slack, or whose client.registration does not
    list managed, refuses it. No response carries a token or a secret.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack.
        body (ConnectorProviderAppRequest): The app the router creates and keeps in the customer's
            workspace. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorProviderApp | ErrorResponse
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
    body: ConnectorProviderAppRequest,
) -> Response[ConnectorProviderApp | ErrorResponse]:
    """Create or update the app the router keeps at the connector's provider

     Creates the customer's own Slack app in its workspace with Slack's apps.manifest.create, from the
    connector's scopes and events, the name given and this router's callback and events URLs, with token
    rotation on. The app's client is what every later consent of the connector's connections uses. It
    needs an app configuration token's refresh token the first time, which a workspace admin generates
    in Slack's app settings; the router rotates it before it expires and keeps it sealed. Putting it
    again changes nothing at Slack but the app's manifest: there is one app per customer and connector,
    never a second. A connector that does not authorize at Slack, or whose client.registration does not
    list managed, refuses it. No response carries a token or a secret.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack.
        body (ConnectorProviderAppRequest): The app the router creates and keeps in the customer's
            workspace. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorProviderApp | ErrorResponse]
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
    body: ConnectorProviderAppRequest,
) -> ConnectorProviderApp | ErrorResponse | None:
    """Create or update the app the router keeps at the connector's provider

     Creates the customer's own Slack app in its workspace with Slack's apps.manifest.create, from the
    connector's scopes and events, the name given and this router's callback and events URLs, with token
    rotation on. The app's client is what every later consent of the connector's connections uses. It
    needs an app configuration token's refresh token the first time, which a workspace admin generates
    in Slack's app settings; the router rotates it before it expires and keeps it sealed. Putting it
    again changes nothing at Slack but the app's manifest: there is one app per customer and connector,
    never a second. A connector that does not authorize at Slack, or whose client.registration does not
    list managed, refuses it. No response carries a token or a secret.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack.
        body (ConnectorProviderAppRequest): The app the router creates and keeps in the customer's
            workspace. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorProviderApp | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
