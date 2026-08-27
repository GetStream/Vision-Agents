from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.call_event import CallEvent
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    limit: int | Unset = 1000,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/calls/{id}/events".format(
            id=quote(str(id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[CallEvent] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = CallEvent.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[CallEvent]]:
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
    limit: int | Unset = 1000,
) -> Response[Error | list[CallEvent]]:
    """What the conversation decided, and why

     A timeline says what a call cost the caller in waiting. This says why the call went the way it did:
    why the agent waited rather than answering, why it read something as not meant for it, why it
    stopped mid-sentence. Read in order they are the reasoning behind the conversation, which is the
    only thing that explains a call that surprised somebody. A call still running reports the same
    decisions live on the session socket.

    Args:
        id (str):
        limit (int | Unset):  Default: 1000.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[CallEvent]]
    """

    kwargs = _get_kwargs(
        id=id,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 1000,
) -> Error | list[CallEvent] | None:
    """What the conversation decided, and why

     A timeline says what a call cost the caller in waiting. This says why the call went the way it did:
    why the agent waited rather than answering, why it read something as not meant for it, why it
    stopped mid-sentence. Read in order they are the reasoning behind the conversation, which is the
    only thing that explains a call that surprised somebody. A call still running reports the same
    decisions live on the session socket.

    Args:
        id (str):
        limit (int | Unset):  Default: 1000.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[CallEvent]
    """

    return sync_detailed(
        id=id,
        client=client,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 1000,
) -> Response[Error | list[CallEvent]]:
    """What the conversation decided, and why

     A timeline says what a call cost the caller in waiting. This says why the call went the way it did:
    why the agent waited rather than answering, why it read something as not meant for it, why it
    stopped mid-sentence. Read in order they are the reasoning behind the conversation, which is the
    only thing that explains a call that surprised somebody. A call still running reports the same
    decisions live on the session socket.

    Args:
        id (str):
        limit (int | Unset):  Default: 1000.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[CallEvent]]
    """

    kwargs = _get_kwargs(
        id=id,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 1000,
) -> Error | list[CallEvent] | None:
    """What the conversation decided, and why

     A timeline says what a call cost the caller in waiting. This says why the call went the way it did:
    why the agent waited rather than answering, why it read something as not meant for it, why it
    stopped mid-sentence. Read in order they are the reasoning behind the conversation, which is the
    only thing that explains a call that surprised somebody. A call still running reports the same
    decisions live on the session socket.

    Args:
        id (str):
        limit (int | Unset):  Default: 1000.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[CallEvent]
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            limit=limit,
        )
    ).parsed
