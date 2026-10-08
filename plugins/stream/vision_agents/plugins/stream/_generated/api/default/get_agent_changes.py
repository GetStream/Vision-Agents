from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_changes import AgentChanges
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/configs/{id}/changes".format(
            id=quote(str(id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentChanges | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = AgentChanges.from_dict(response.json())

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
) -> Response[AgentChanges | ErrorResponse]:
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
) -> Response[AgentChanges | ErrorResponse]:
    """What was changed about an agent since its directory was last synced

     The edits a sync of the agent's directory would write over: everything changed about the agent, its
    skills and its knowledge since the last sync, newest first. An agent nobody has touched since
    answers with an empty list.

    This is what a sync refused with `unsynced_changes` is asking about. Show the changes, let the
    person decide, and sync again with `base_change` set to `last_change` to say they have been seen --
    having either written them into the directory first, or chosen to write over them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The agent config, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentChanges | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> AgentChanges | ErrorResponse | None:
    """What was changed about an agent since its directory was last synced

     The edits a sync of the agent's directory would write over: everything changed about the agent, its
    skills and its knowledge since the last sync, newest first. An agent nobody has touched since
    answers with an empty list.

    This is what a sync refused with `unsynced_changes` is asking about. Show the changes, let the
    person decide, and sync again with `base_change` set to `last_change` to say they have been seen --
    having either written them into the directory first, or chosen to write over them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The agent config, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentChanges | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[AgentChanges | ErrorResponse]:
    """What was changed about an agent since its directory was last synced

     The edits a sync of the agent's directory would write over: everything changed about the agent, its
    skills and its knowledge since the last sync, newest first. An agent nobody has touched since
    answers with an empty list.

    This is what a sync refused with `unsynced_changes` is asking about. Show the changes, let the
    person decide, and sync again with `base_change` set to `last_change` to say they have been seen --
    having either written them into the directory first, or chosen to write over them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The agent config, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentChanges | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> AgentChanges | ErrorResponse | None:
    """What was changed about an agent since its directory was last synced

     The edits a sync of the agent's directory would write over: everything changed about the agent, its
    skills and its knowledge since the last sync, newest first. An agent nobody has touched since
    answers with an empty list.

    This is what a sync refused with `unsynced_changes` is asking about. Show the changes, let the
    person decide, and sync again with `base_change` set to `last_change` to say they have been seen --
    having either written them into the directory first, or chosen to write over them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The agent config, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentChanges | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
