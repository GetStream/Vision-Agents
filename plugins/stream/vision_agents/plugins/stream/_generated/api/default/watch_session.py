from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["interim"] = interim

    params["decisions"] = decisions

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/sessions/{id}/events".format(
            id=quote(str(id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | Error | None:
    if response.status_code == 101:
        response_101 = cast(Any, None)
        return response_101

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
) -> Response[Any | Error]:
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
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
) -> Response[Any | Error]:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded`, `spoke`, `turn`, `decision`, `delegated`, `task_settled`, `task_cancelled`,
    `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`, `backchannel`, `interrupted`,
    `overlap_decided`, `conversation_compacted`, `error` and `left`.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`,
    `instructions` or `close` to act on the session. A `tool_call` is the only frame that must be
    answered: everything else is a report.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        interim=interim,
        decisions=decisions,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
) -> Any | Error | None:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded`, `spoke`, `turn`, `decision`, `delegated`, `task_settled`, `task_cancelled`,
    `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`, `backchannel`, `interrupted`,
    `overlap_decided`, `conversation_compacted`, `error` and `left`.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`,
    `instructions` or `close` to act on the session. A `tool_call` is the only frame that must be
    answered: everything else is a report.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error
    """

    return sync_detailed(
        id=id,
        client=client,
        interim=interim,
        decisions=decisions,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
) -> Response[Any | Error]:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded`, `spoke`, `turn`, `decision`, `delegated`, `task_settled`, `task_cancelled`,
    `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`, `backchannel`, `interrupted`,
    `overlap_decided`, `conversation_compacted`, `error` and `left`.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`,
    `instructions` or `close` to act on the session. A `tool_call` is the only frame that must be
    answered: everything else is a report.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        interim=interim,
        decisions=decisions,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
) -> Any | Error | None:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded`, `spoke`, `turn`, `decision`, `delegated`, `task_settled`, `task_cancelled`,
    `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`, `backchannel`, `interrupted`,
    `overlap_decided`, `conversation_compacted`, `error` and `left`.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`,
    `instructions` or `close` to act on the session. A `tool_call` is the only frame that must be
    answered: everything else is a report.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            interim=interim,
            decisions=decisions,
        )
    ).parsed
