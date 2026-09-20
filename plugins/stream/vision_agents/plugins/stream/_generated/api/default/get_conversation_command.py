from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.command_receipt import CommandReceipt
from ...models.error import Error
from ...types import UNSET, Response


def _get_kwargs(
    cid: str,
    command_id: str,
    *,
    agent_id: str,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["agent_id"] = agent_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/conversations/{cid}/commands/{command_id}".format(
            cid=quote(str(cid), safe=""),
            command_id=quote(str(command_id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> CommandReceipt | Error | None:
    if response.status_code == 200:
        response_200 = CommandReceipt.from_dict(response.json())

        return response_200

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
) -> Response[CommandReceipt | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    cid: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
) -> Response[CommandReceipt | Error]:
    """What a command in this conversation ended as

     Reads one command's receipt from the conversation's own durable record. It opens nothing and starts
    nothing, so a client whose stop found no session left to reach reconciles that command here rather
    than reopening a session to ask about it.
    A command still running is reported as it stands; the session holding it is where it can be stopped.

    Args:
        cid (str):
        command_id (str):
        agent_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CommandReceipt | Error]
    """

    kwargs = _get_kwargs(
        cid=cid,
        command_id=command_id,
        agent_id=agent_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    cid: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
) -> CommandReceipt | Error | None:
    """What a command in this conversation ended as

     Reads one command's receipt from the conversation's own durable record. It opens nothing and starts
    nothing, so a client whose stop found no session left to reach reconciles that command here rather
    than reopening a session to ask about it.
    A command still running is reported as it stands; the session holding it is where it can be stopped.

    Args:
        cid (str):
        command_id (str):
        agent_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CommandReceipt | Error
    """

    return sync_detailed(
        cid=cid,
        command_id=command_id,
        client=client,
        agent_id=agent_id,
    ).parsed


async def asyncio_detailed(
    cid: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
) -> Response[CommandReceipt | Error]:
    """What a command in this conversation ended as

     Reads one command's receipt from the conversation's own durable record. It opens nothing and starts
    nothing, so a client whose stop found no session left to reach reconciles that command here rather
    than reopening a session to ask about it.
    A command still running is reported as it stands; the session holding it is where it can be stopped.

    Args:
        cid (str):
        command_id (str):
        agent_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CommandReceipt | Error]
    """

    kwargs = _get_kwargs(
        cid=cid,
        command_id=command_id,
        agent_id=agent_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    cid: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
    agent_id: str,
) -> CommandReceipt | Error | None:
    """What a command in this conversation ended as

     Reads one command's receipt from the conversation's own durable record. It opens nothing and starts
    nothing, so a client whose stop found no session left to reach reconciles that command here rather
    than reopening a session to ask about it.
    A command still running is reported as it stands; the session holding it is where it can be stopped.

    Args:
        cid (str):
        command_id (str):
        agent_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CommandReceipt | Error
    """

    return (
        await asyncio_detailed(
            cid=cid,
            command_id=command_id,
            client=client,
            agent_id=agent_id,
        )
    ).parsed
