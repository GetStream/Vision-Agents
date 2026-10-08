from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.channel_link import ChannelLink
from ...models.error_response import ErrorResponse
from ...models.link_channel_request import LinkChannelRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: LinkChannelRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/channels/links",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ChannelLink | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ChannelLink.from_dict(response.json())

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
) -> Response[ChannelLink | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: LinkChannelRequest | Unset = UNSET,
) -> Response[ChannelLink | ErrorResponse]:
    """Mint a code to claim a number

     Answers with a code to show somebody already signed in. The number that texts it to the agent
    belongs to that end user from then on, which is what an agent reading a person's own calendar or
    orders needs before it says a word. Only an agent whose `channels.identity` is `link` has anything
    to link.

    Server-side only.

    Args:
        body (LinkChannelRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ChannelLink | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: LinkChannelRequest | Unset = UNSET,
) -> ChannelLink | ErrorResponse | None:
    """Mint a code to claim a number

     Answers with a code to show somebody already signed in. The number that texts it to the agent
    belongs to that end user from then on, which is what an agent reading a person's own calendar or
    orders needs before it says a word. Only an agent whose `channels.identity` is `link` has anything
    to link.

    Server-side only.

    Args:
        body (LinkChannelRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ChannelLink | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: LinkChannelRequest | Unset = UNSET,
) -> Response[ChannelLink | ErrorResponse]:
    """Mint a code to claim a number

     Answers with a code to show somebody already signed in. The number that texts it to the agent
    belongs to that end user from then on, which is what an agent reading a person's own calendar or
    orders needs before it says a word. Only an agent whose `channels.identity` is `link` has anything
    to link.

    Server-side only.

    Args:
        body (LinkChannelRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ChannelLink | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: LinkChannelRequest | Unset = UNSET,
) -> ChannelLink | ErrorResponse | None:
    """Mint a code to claim a number

     Answers with a code to show somebody already signed in. The number that texts it to the agent
    belongs to that end user from then on, which is what an agent reading a person's own calendar or
    orders needs before it says a word. Only an agent whose `channels.identity` is `link` has anything
    to link.

    Server-side only.

    Args:
        body (LinkChannelRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ChannelLink | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
