from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    destination_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/agents/connectors/{id}/event-destinations/{destination_id}".format(
            id=quote(str(id), safe=""),
            destination_id=quote(str(destination_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

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
) -> Response[Any | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Stop forwarding to an event destination

     Removes the destination, and every forward to it not yet sent.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        destination_id=destination_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Stop forwarding to an event destination

     Removes the destination, and every forward to it not yet sent.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        id=id,
        destination_id=destination_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Stop forwarding to an event destination

     Removes the destination, and every forward to it not yet sent.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        destination_id=destination_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Stop forwarding to an event destination

     Removes the destination, and every forward to it not yet sent.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            destination_id=destination_id,
            client=client,
        )
    ).parsed
