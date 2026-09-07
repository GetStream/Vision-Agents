from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.chat_token import ChatToken
from ...models.chat_token_request import ChatTokenRequest
from ...models.error import Error
from ...types import Response


def _get_kwargs(
    *,
    body: ChatTokenRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/chat-token",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ChatToken | Error | None:
    if response.status_code == 200:
        response_200 = ChatToken.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ChatToken | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ChatTokenRequest,
) -> Response[ChatToken | Error]:
    """What a browser needs to read an agent's conversation

     An agent writes what was said into the Stream Chat channel agent:{agent_id}, so a client that can
    read that channel needs no transcript API. This mints the token to read it with, and adds the reader
    to the channel, since a conversation they are not a member of is one they cannot watch.
    The secret stays here, the same as for a call token: the browser is handed something that expires.

    Args:
        body (ChatTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ChatToken | Error]
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
    body: ChatTokenRequest,
) -> ChatToken | Error | None:
    """What a browser needs to read an agent's conversation

     An agent writes what was said into the Stream Chat channel agent:{agent_id}, so a client that can
    read that channel needs no transcript API. This mints the token to read it with, and adds the reader
    to the channel, since a conversation they are not a member of is one they cannot watch.
    The secret stays here, the same as for a call token: the browser is handed something that expires.

    Args:
        body (ChatTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ChatToken | Error
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ChatTokenRequest,
) -> Response[ChatToken | Error]:
    """What a browser needs to read an agent's conversation

     An agent writes what was said into the Stream Chat channel agent:{agent_id}, so a client that can
    read that channel needs no transcript API. This mints the token to read it with, and adds the reader
    to the channel, since a conversation they are not a member of is one they cannot watch.
    The secret stays here, the same as for a call token: the browser is handed something that expires.

    Args:
        body (ChatTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ChatToken | Error]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ChatTokenRequest,
) -> ChatToken | Error | None:
    """What a browser needs to read an agent's conversation

     An agent writes what was said into the Stream Chat channel agent:{agent_id}, so a client that can
    read that channel needs no transcript API. This mints the token to read it with, and adds the reader
    to the channel, since a conversation they are not a member of is one they cannot watch.
    The secret stays here, the same as for a call token: the browser is handed something that expires.

    Args:
        body (ChatTokenRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ChatToken | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
