from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_response_page import AgentResponsePage
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/sessions/{id}/responses".format(
            id=quote(str(id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentResponsePage | Error | None:
    if response.status_code == 200:
        response_200 = AgentResponsePage.from_dict(response.json())

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
) -> Response[AgentResponsePage | Error]:
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
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[AgentResponsePage | Error]:
    """The turns the agent took in a session

     Oldest first, which read in order are the conversation. This is the shape of it rather than the
    text: what was asked, whether the turn finished, and how long it took. The items endpoint is what
    carries what happened inside each one.

    Args:
        id (str):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentResponsePage | Error]
    """

    kwargs = _get_kwargs(
        id=id,
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
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> AgentResponsePage | Error | None:
    """The turns the agent took in a session

     Oldest first, which read in order are the conversation. This is the shape of it rather than the
    text: what was asked, whether the turn finished, and how long it took. The items endpoint is what
    carries what happened inside each one.

    Args:
        id (str):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentResponsePage | Error
    """

    return sync_detailed(
        id=id,
        client=client,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[AgentResponsePage | Error]:
    """The turns the agent took in a session

     Oldest first, which read in order are the conversation. This is the shape of it rather than the
    text: what was asked, whether the turn finished, and how long it took. The items endpoint is what
    carries what happened inside each one.

    Args:
        id (str):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentResponsePage | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> AgentResponsePage | Error | None:
    """The turns the agent took in a session

     Oldest first, which read in order are the conversation. This is the shape of it rather than the
    text: what was asked, whether the turn finished, and how long it took. The items endpoint is what
    carries what happened inside each one.

    Args:
        id (str):
        limit (int | Unset):
        cursor (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentResponsePage | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
