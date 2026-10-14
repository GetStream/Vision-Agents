from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    plugin_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/plugins/{plugin_id}/logo".format(
            plugin_id=quote(str(plugin_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | None:
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
) -> Response[ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ErrorResponse]:
    """A plugin's logo

     Deprecated with the plugin catalog. Connectors have no logo route. The image a card uses to show
    which plugin it is asking about, as an SVG. The path is unauthenticated because what draws it is an
    `<img>` in a chat client or a browser, which has no credential of this API's to send, and because
    the catalog is the same built-in list for every customer, so there is nothing of anybody's here.

    Args:
        plugin_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse]
    """

    kwargs = _get_kwargs(
        plugin_id=plugin_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | None:
    """A plugin's logo

     Deprecated with the plugin catalog. Connectors have no logo route. The image a card uses to show
    which plugin it is asking about, as an SVG. The path is unauthenticated because what draws it is an
    `<img>` in a chat client or a browser, which has no credential of this API's to send, and because
    the catalog is the same built-in list for every customer, so there is nothing of anybody's here.

    Args:
        plugin_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse
    """

    return sync_detailed(
        plugin_id=plugin_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ErrorResponse]:
    """A plugin's logo

     Deprecated with the plugin catalog. Connectors have no logo route. The image a card uses to show
    which plugin it is asking about, as an SVG. The path is unauthenticated because what draws it is an
    `<img>` in a chat client or a browser, which has no credential of this API's to send, and because
    the catalog is the same built-in list for every customer, so there is nothing of anybody's here.

    Args:
        plugin_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse]
    """

    kwargs = _get_kwargs(
        plugin_id=plugin_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    plugin_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | None:
    """A plugin's logo

     Deprecated with the plugin catalog. Connectors have no logo route. The image a card uses to show
    which plugin it is asking about, as an SVG. The path is unauthenticated because what draws it is an
    `<img>` in a chat client or a browser, which has no credential of this API's to send, and because
    the catalog is the same built-in list for every customer, so there is nothing of anybody's here.

    Args:
        plugin_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse
    """

    return (
        await asyncio_detailed(
            plugin_id=plugin_id,
            client=client,
        )
    ).parsed
