from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_response_item_page import AgentResponseItemPage
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    response_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["response_id"] = response_id

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/sessions/{id}/responses/items".format(
            id=quote(str(id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentResponseItemPage | Error | None:
    if response.status_code == 200:
        response_200 = AgentResponseItemPage.from_dict(response.json())

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

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[AgentResponseItemPage | Error]:
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
    response_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[AgentResponseItemPage | Error]:
    """What the agent did, turn by turn, in the order it happened

     One flat stream across every turn rather than a list per turn, because that is how a conversation
    reads and how it is rendered: the question, what the agent did about it, what it said, then the next
    question. Naming a response narrows it to that turn.
    Deltas are not here. A hundred fragments of one sentence are the sentence, and keeping them would
    make this mostly punctuation; a caller watching a turn happen reads the deltas off the events
    socket, and a caller reading one back wants the shape of it.
    Nothing is returned for an incognito session, which has no items to return.

    Args:
        id (str):
        response_id (str | Unset):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentResponseItemPage | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        response_id=response_id,
        limit=limit,
        cursor=cursor,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    response_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> AgentResponseItemPage | Error | None:
    """What the agent did, turn by turn, in the order it happened

     One flat stream across every turn rather than a list per turn, because that is how a conversation
    reads and how it is rendered: the question, what the agent did about it, what it said, then the next
    question. Naming a response narrows it to that turn.
    Deltas are not here. A hundred fragments of one sentence are the sentence, and keeping them would
    make this mostly punctuation; a caller watching a turn happen reads the deltas off the events
    socket, and a caller reading one back wants the shape of it.
    Nothing is returned for an incognito session, which has no items to return.

    Args:
        id (str):
        response_id (str | Unset):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentResponseItemPage | Error
    """

    return sync_detailed(
        id=id,
        client=client,
        response_id=response_id,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    response_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[AgentResponseItemPage | Error]:
    """What the agent did, turn by turn, in the order it happened

     One flat stream across every turn rather than a list per turn, because that is how a conversation
    reads and how it is rendered: the question, what the agent did about it, what it said, then the next
    question. Naming a response narrows it to that turn.
    Deltas are not here. A hundred fragments of one sentence are the sentence, and keeping them would
    make this mostly punctuation; a caller watching a turn happen reads the deltas off the events
    socket, and a caller reading one back wants the shape of it.
    Nothing is returned for an incognito session, which has no items to return.

    Args:
        id (str):
        response_id (str | Unset):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentResponseItemPage | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        response_id=response_id,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    response_id: str | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> AgentResponseItemPage | Error | None:
    """What the agent did, turn by turn, in the order it happened

     One flat stream across every turn rather than a list per turn, because that is how a conversation
    reads and how it is rendered: the question, what the agent did about it, what it said, then the next
    question. Naming a response narrows it to that turn.
    Deltas are not here. A hundred fragments of one sentence are the sentence, and keeping them would
    make this mostly punctuation; a caller watching a turn happen reads the deltas off the events
    socket, and a caller reading one back wants the shape of it.
    Nothing is returned for an incognito session, which has no items to return.

    Args:
        id (str):
        response_id (str | Unset):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentResponseItemPage | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            response_id=response_id,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
