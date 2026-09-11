from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.get_conversation_messages_response_200 import (
    GetConversationMessagesResponse200,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    cid: str,
    *,
    agent_id: str,
    before: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["agent_id"] = agent_id

    params["before"] = before

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/conversations/{cid}/messages".format(
            cid=quote(str(cid), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | GetConversationMessagesResponse200 | None:
    if response.status_code == 200:
        response_200 = GetConversationMessagesResponse200.from_dict(response.json())

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
) -> Response[Error | GetConversationMessagesResponse200]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
    before: str | Unset = UNSET,
) -> Response[Error | GetConversationMessagesResponse200]:
    """Read a persistent text conversation

    Args:
        cid (str):
        agent_id (str):
        before (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | GetConversationMessagesResponse200]
    """

    kwargs = _get_kwargs(
        cid=cid,
        agent_id=agent_id,
        before=before,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
    before: str | Unset = UNSET,
) -> Error | GetConversationMessagesResponse200 | None:
    """Read a persistent text conversation

    Args:
        cid (str):
        agent_id (str):
        before (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | GetConversationMessagesResponse200
    """

    return sync_detailed(
        cid=cid,
        client=client,
        agent_id=agent_id,
        before=before,
    ).parsed


async def asyncio_detailed(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
    before: str | Unset = UNSET,
) -> Response[Error | GetConversationMessagesResponse200]:
    """Read a persistent text conversation

    Args:
        cid (str):
        agent_id (str):
        before (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | GetConversationMessagesResponse200]
    """

    kwargs = _get_kwargs(
        cid=cid,
        agent_id=agent_id,
        before=before,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
    before: str | Unset = UNSET,
) -> Error | GetConversationMessagesResponse200 | None:
    """Read a persistent text conversation

    Args:
        cid (str):
        agent_id (str):
        before (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | GetConversationMessagesResponse200
    """

    return (
        await asyncio_detailed(
            cid=cid,
            client=client,
            agent_id=agent_id,
            before=before,
        )
    ).parsed
