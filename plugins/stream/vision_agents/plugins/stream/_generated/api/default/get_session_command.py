from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.command_receipt import CommandReceipt
from ...models.error import Error
from ...types import Response


def _get_kwargs(
    id: str,
    command_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/sessions/{id}/commands/{command_id}".format(
            id=quote(str(id), safe=""),
            command_id=quote(str(command_id), safe=""),
        ),
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
    id: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[CommandReceipt | Error]:
    """What is known about one durable command

     Reads a command's receipt without accepting, running or stopping anything. It is how a client whose
    stop or submission had an unknown outcome reconciles the same command id rather than inventing
    another one.

    Args:
        id (str):
        command_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CommandReceipt | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        command_id=command_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> CommandReceipt | Error | None:
    """What is known about one durable command

     Reads a command's receipt without accepting, running or stopping anything. It is how a client whose
    stop or submission had an unknown outcome reconciles the same command id rather than inventing
    another one.

    Args:
        id (str):
        command_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CommandReceipt | Error
    """

    return sync_detailed(
        id=id,
        command_id=command_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[CommandReceipt | Error]:
    """What is known about one durable command

     Reads a command's receipt without accepting, running or stopping anything. It is how a client whose
    stop or submission had an unknown outcome reconciles the same command id rather than inventing
    another one.

    Args:
        id (str):
        command_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CommandReceipt | Error]
    """

    kwargs = _get_kwargs(
        id=id,
        command_id=command_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    command_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> CommandReceipt | Error | None:
    """What is known about one durable command

     Reads a command's receipt without accepting, running or stopping anything. It is how a client whose
    stop or submission had an unknown outcome reconciles the same command id rather than inventing
    another one.

    Args:
        id (str):
        command_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CommandReceipt | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            command_id=command_id,
            client=client,
        )
    ).parsed
