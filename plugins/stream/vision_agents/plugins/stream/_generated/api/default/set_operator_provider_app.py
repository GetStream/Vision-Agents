from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_provider_app import ConnectorProviderApp
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    customer_id: str,
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/ops/customers/{customer_id}/connectors/{id}/provider-app".format(
            customer_id=quote(str(customer_id), safe=""),
            id=quote(str(id), safe=""),
        ),
    }

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
) -> Response[ConnectorProviderApp | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    customer_id: str,
    id: str,
    *,
    client: AuthenticatedClient,
) -> Response[ConnectorProviderApp | ErrorResponse]:
    """Make Stream's own app a customer's provider app

     Records this deployment's own app for a built-in connector, as its environment holds it
    (<client.env>_MCP_APP_ID, _MCP_CLIENT_ID, _MCP_CLIENT_SECRET and _MCP_SIGNING_SECRET), as the
    customer's provider app, so its events reach that customer. One customer per app: another customer's
    record of it is a conflict.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The customer Stream's app serves.
        id (str): The built-in connector, such as slack.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorProviderApp | ErrorResponse]
    """

    kwargs = _get_kwargs(
        customer_id=customer_id,
        id=id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    customer_id: str,
    id: str,
    *,
    client: AuthenticatedClient,
) -> ConnectorProviderApp | ErrorResponse | None:
    """Make Stream's own app a customer's provider app

     Records this deployment's own app for a built-in connector, as its environment holds it
    (<client.env>_MCP_APP_ID, _MCP_CLIENT_ID, _MCP_CLIENT_SECRET and _MCP_SIGNING_SECRET), as the
    customer's provider app, so its events reach that customer. One customer per app: another customer's
    record of it is a conflict.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The customer Stream's app serves.
        id (str): The built-in connector, such as slack.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorProviderApp | ErrorResponse
    """

    return sync_detailed(
        customer_id=customer_id,
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    customer_id: str,
    id: str,
    *,
    client: AuthenticatedClient,
) -> Response[ConnectorProviderApp | ErrorResponse]:
    """Make Stream's own app a customer's provider app

     Records this deployment's own app for a built-in connector, as its environment holds it
    (<client.env>_MCP_APP_ID, _MCP_CLIENT_ID, _MCP_CLIENT_SECRET and _MCP_SIGNING_SECRET), as the
    customer's provider app, so its events reach that customer. One customer per app: another customer's
    record of it is a conflict.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The customer Stream's app serves.
        id (str): The built-in connector, such as slack.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorProviderApp | ErrorResponse]
    """

    kwargs = _get_kwargs(
        customer_id=customer_id,
        id=id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    customer_id: str,
    id: str,
    *,
    client: AuthenticatedClient,
) -> ConnectorProviderApp | ErrorResponse | None:
    """Make Stream's own app a customer's provider app

     Records this deployment's own app for a built-in connector, as its environment holds it
    (<client.env>_MCP_APP_ID, _MCP_CLIENT_ID, _MCP_CLIENT_SECRET and _MCP_SIGNING_SECRET), as the
    customer's provider app, so its events reach that customer. One customer per app: another customer's
    record of it is a conflict.

    When the provider app is pinned to a Stream app the customer registered, the router then points that
    app's message hook at itself, at ROUTER_PUBLIC_URL/v1/chat/hooks/stream/{stream app id}: it adds the
    hook, or updates the one already there, so the messages written in the app's thread channels reach
    the router. A router without ROUTER_PUBLIC_URL points none and logs a warning. When Stream refuses,
    the provider app is kept, the answer is a 503, and putting it again points the hook again.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The customer Stream's app serves.
        id (str): The built-in connector, such as slack.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorProviderApp | ErrorResponse
    """

    return (
        await asyncio_detailed(
            customer_id=customer_id,
            id=id,
            client=client,
        )
    ).parsed
