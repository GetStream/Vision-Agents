import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.search_sessions_state import SearchSessionsState
from ...models.session import Session
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    q: str | Unset = UNSET,
    agent: str | Unset = UNSET,
    config_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    project: str | Unset = UNSET,
    state: SearchSessionsState | Unset = UNSET,
    custom: str | Unset = UNSET,
    created_after: datetime.datetime | Unset = UNSET,
    created_before: datetime.datetime | Unset = UNSET,
    limit: int | Unset = UNSET,
    offset: int | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["q"] = q

    params["agent"] = agent

    params["config_id"] = config_id

    params["user_id"] = user_id

    params["project"] = project

    json_state: str | Unset = UNSET
    if not isinstance(state, Unset):
        json_state = state.value

    params["state"] = json_state

    params["custom"] = custom

    json_created_after: str | Unset = UNSET
    if not isinstance(created_after, Unset):
        json_created_after = created_after.isoformat()
    params["created_after"] = json_created_after

    json_created_before: str | Unset = UNSET
    if not isinstance(created_before, Unset):
        json_created_before = created_before.isoformat()
    params["created_before"] = json_created_before

    params["limit"] = limit

    params["offset"] = offset

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/sessions/search",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[Session] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Session.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Error | list[Session]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
    agent: str | Unset = UNSET,
    config_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    project: str | Unset = UNSET,
    state: SearchSessionsState | Unset = UNSET,
    custom: str | Unset = UNSET,
    created_after: datetime.datetime | Unset = UNSET,
    created_before: datetime.datetime | Unset = UNSET,
    limit: int | Unset = UNSET,
    offset: int | Unset = UNSET,
) -> Response[Error | list[Session]]:
    """Find a conversation by what it was called

     Full text over the title, description, project and agent name, best match first, with titles
    weighted above the rest so the conversation called "billing" beats every conversation in the billing
    project.
    What was said is not searched. Doing so would mean either reading every conversation out of Stream
    Chat on each query, which is too slow to offer, or keeping a second copy of every message here,
    which is a transcript that can drift from the real one. Titles and descriptions are what a person
    names a conversation with, and naming them is the habit worth encouraging.
    The same owner scoping as listing applies, and the same filters narrow it, so a search cannot reach
    a conversation a list could not. An empty q is the same as no q and falls through to the list,
    because a search box nobody has typed in yet should show a person their conversations rather than
    nothing.

    Args:
        q (str | Unset):
        agent (str | Unset):
        config_id (str | Unset):
        user_id (str | Unset):
        project (str | Unset):
        state (SearchSessionsState | Unset):
        custom (str | Unset):
        created_after (datetime.datetime | Unset):
        created_before (datetime.datetime | Unset):
        limit (int | Unset):
        offset (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Session]]
    """

    kwargs = _get_kwargs(
        q=q,
        agent=agent,
        config_id=config_id,
        user_id=user_id,
        project=project,
        state=state,
        custom=custom,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
    agent: str | Unset = UNSET,
    config_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    project: str | Unset = UNSET,
    state: SearchSessionsState | Unset = UNSET,
    custom: str | Unset = UNSET,
    created_after: datetime.datetime | Unset = UNSET,
    created_before: datetime.datetime | Unset = UNSET,
    limit: int | Unset = UNSET,
    offset: int | Unset = UNSET,
) -> Error | list[Session] | None:
    """Find a conversation by what it was called

     Full text over the title, description, project and agent name, best match first, with titles
    weighted above the rest so the conversation called "billing" beats every conversation in the billing
    project.
    What was said is not searched. Doing so would mean either reading every conversation out of Stream
    Chat on each query, which is too slow to offer, or keeping a second copy of every message here,
    which is a transcript that can drift from the real one. Titles and descriptions are what a person
    names a conversation with, and naming them is the habit worth encouraging.
    The same owner scoping as listing applies, and the same filters narrow it, so a search cannot reach
    a conversation a list could not. An empty q is the same as no q and falls through to the list,
    because a search box nobody has typed in yet should show a person their conversations rather than
    nothing.

    Args:
        q (str | Unset):
        agent (str | Unset):
        config_id (str | Unset):
        user_id (str | Unset):
        project (str | Unset):
        state (SearchSessionsState | Unset):
        custom (str | Unset):
        created_after (datetime.datetime | Unset):
        created_before (datetime.datetime | Unset):
        limit (int | Unset):
        offset (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Session]
    """

    return sync_detailed(
        client=client,
        q=q,
        agent=agent,
        config_id=config_id,
        user_id=user_id,
        project=project,
        state=state,
        custom=custom,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
    agent: str | Unset = UNSET,
    config_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    project: str | Unset = UNSET,
    state: SearchSessionsState | Unset = UNSET,
    custom: str | Unset = UNSET,
    created_after: datetime.datetime | Unset = UNSET,
    created_before: datetime.datetime | Unset = UNSET,
    limit: int | Unset = UNSET,
    offset: int | Unset = UNSET,
) -> Response[Error | list[Session]]:
    """Find a conversation by what it was called

     Full text over the title, description, project and agent name, best match first, with titles
    weighted above the rest so the conversation called "billing" beats every conversation in the billing
    project.
    What was said is not searched. Doing so would mean either reading every conversation out of Stream
    Chat on each query, which is too slow to offer, or keeping a second copy of every message here,
    which is a transcript that can drift from the real one. Titles and descriptions are what a person
    names a conversation with, and naming them is the habit worth encouraging.
    The same owner scoping as listing applies, and the same filters narrow it, so a search cannot reach
    a conversation a list could not. An empty q is the same as no q and falls through to the list,
    because a search box nobody has typed in yet should show a person their conversations rather than
    nothing.

    Args:
        q (str | Unset):
        agent (str | Unset):
        config_id (str | Unset):
        user_id (str | Unset):
        project (str | Unset):
        state (SearchSessionsState | Unset):
        custom (str | Unset):
        created_after (datetime.datetime | Unset):
        created_before (datetime.datetime | Unset):
        limit (int | Unset):
        offset (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Session]]
    """

    kwargs = _get_kwargs(
        q=q,
        agent=agent,
        config_id=config_id,
        user_id=user_id,
        project=project,
        state=state,
        custom=custom,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
    agent: str | Unset = UNSET,
    config_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    project: str | Unset = UNSET,
    state: SearchSessionsState | Unset = UNSET,
    custom: str | Unset = UNSET,
    created_after: datetime.datetime | Unset = UNSET,
    created_before: datetime.datetime | Unset = UNSET,
    limit: int | Unset = UNSET,
    offset: int | Unset = UNSET,
) -> Error | list[Session] | None:
    """Find a conversation by what it was called

     Full text over the title, description, project and agent name, best match first, with titles
    weighted above the rest so the conversation called "billing" beats every conversation in the billing
    project.
    What was said is not searched. Doing so would mean either reading every conversation out of Stream
    Chat on each query, which is too slow to offer, or keeping a second copy of every message here,
    which is a transcript that can drift from the real one. Titles and descriptions are what a person
    names a conversation with, and naming them is the habit worth encouraging.
    The same owner scoping as listing applies, and the same filters narrow it, so a search cannot reach
    a conversation a list could not. An empty q is the same as no q and falls through to the list,
    because a search box nobody has typed in yet should show a person their conversations rather than
    nothing.

    Args:
        q (str | Unset):
        agent (str | Unset):
        config_id (str | Unset):
        user_id (str | Unset):
        project (str | Unset):
        state (SearchSessionsState | Unset):
        custom (str | Unset):
        created_after (datetime.datetime | Unset):
        created_before (datetime.datetime | Unset):
        limit (int | Unset):
        offset (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Session]
    """

    return (
        await asyncio_detailed(
            client=client,
            q=q,
            agent=agent,
            config_id=config_id,
            user_id=user_id,
            project=project,
            state=state,
            custom=custom,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            offset=offset,
        )
    ).parsed
