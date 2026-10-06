from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.channel_account import ChannelAccount
from ...models.connect_channel_request import ConnectChannelRequest
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: ConnectChannelRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/channels",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ChannelAccount | Error | None:
    if response.status_code == 200:
        response_200 = ChannelAccount.from_dict(response.json())

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

    if response.status_code == 500:
        response_500 = Error.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ChannelAccount | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectChannelRequest | Unset = UNSET,
) -> Response[ChannelAccount | Error]:
    """Connect a channel line

     Stores an app's credentials for a WhatsApp, text or iMessage line and answers with the URL its
    provider should deliver to. An agent is then reachable there by naming the number under `channels`
    in its `agent.yaml`.

    Sending a line that is already connected replaces its credentials and keeps its webhook URL, so
    rotating a token does not mean setting the webhook up again.

    Server-side only: it carries provider credentials.

    Args:
        body (ConnectChannelRequest | Unset): Credentials for one line, connected once for the app
            and named by any number of agents under channels in agent.yaml. Sending a line already
            connected replaces its credentials and keeps its webhook URL.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ChannelAccount | Error]
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
    body: ConnectChannelRequest | Unset = UNSET,
) -> ChannelAccount | Error | None:
    """Connect a channel line

     Stores an app's credentials for a WhatsApp, text or iMessage line and answers with the URL its
    provider should deliver to. An agent is then reachable there by naming the number under `channels`
    in its `agent.yaml`.

    Sending a line that is already connected replaces its credentials and keeps its webhook URL, so
    rotating a token does not mean setting the webhook up again.

    Server-side only: it carries provider credentials.

    Args:
        body (ConnectChannelRequest | Unset): Credentials for one line, connected once for the app
            and named by any number of agents under channels in agent.yaml. Sending a line already
            connected replaces its credentials and keeps its webhook URL.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ChannelAccount | Error
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectChannelRequest | Unset = UNSET,
) -> Response[ChannelAccount | Error]:
    """Connect a channel line

     Stores an app's credentials for a WhatsApp, text or iMessage line and answers with the URL its
    provider should deliver to. An agent is then reachable there by naming the number under `channels`
    in its `agent.yaml`.

    Sending a line that is already connected replaces its credentials and keeps its webhook URL, so
    rotating a token does not mean setting the webhook up again.

    Server-side only: it carries provider credentials.

    Args:
        body (ConnectChannelRequest | Unset): Credentials for one line, connected once for the app
            and named by any number of agents under channels in agent.yaml. Sending a line already
            connected replaces its credentials and keeps its webhook URL.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ChannelAccount | Error]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectChannelRequest | Unset = UNSET,
) -> ChannelAccount | Error | None:
    """Connect a channel line

     Stores an app's credentials for a WhatsApp, text or iMessage line and answers with the URL its
    provider should deliver to. An agent is then reachable there by naming the number under `channels`
    in its `agent.yaml`.

    Sending a line that is already connected replaces its credentials and keeps its webhook URL, so
    rotating a token does not mean setting the webhook up again.

    Server-side only: it carries provider credentials.

    Args:
        body (ConnectChannelRequest | Unset): Credentials for one line, connected once for the app
            and named by any number of agents under channels in agent.yaml. Sending a line already
            connected replaces its credentials and keeps its webhook URL.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ChannelAccount | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
