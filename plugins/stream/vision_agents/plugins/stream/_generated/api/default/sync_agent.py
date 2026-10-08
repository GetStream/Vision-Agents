from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.sync_agent_request import SyncAgentRequest
from ...models.sync_agent_result import SyncAgentResult
from ...types import Response


def _get_kwargs(
    *,
    body: SyncAgentRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sync",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | SyncAgentResult | None:
    if response.status_code == 200:
        response_200 = SyncAgentResult.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | SyncAgentResult]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SyncAgentRequest,
) -> Response[ErrorResponse | SyncAgentResult]:
    """Store an agent directory's instructions, skills, knowledge, simulations and settings

     Reads as "this is what the agent is", from a directory of agent.yaml, instructions.md, skills/,
    knowledge/ and simulations/. The hash is a fingerprint of that directory: a second call with the
    same hash does nothing, so a process that syncs on startup is cheap when nothing has changed.

    agent.yaml decides the models, the voice and the rest of a config, so an agent kept in a repository
    needs nothing written by hand. A setting it leaves out is left alone rather than blanked.

    knowledge/ is the whole of the knowledge base named after the agent, and simulations/ the whole of
    its simulations: a file taken out of the directory is taken out of the backend on the next sync.

    A directory is not the only thing that writes an agent: somebody may have changed one of the same
    settings in the dashboard since the last sync. Send `check_changes` and such a sync is refused with
    `unsynced_changes` instead of writing over them -- only when it really would write over them, so a
    directory that already holds what the dashboard says syncs without complaint. Read the changes from
    `GET /v1/agents/configs/{id}/changes`, let the person decide, and sync again with `base_change` to
    go ahead.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest): An agent directory as it is on disk. Everything after the
            simulations is what the directory's declaration decides rather than what it holds, and a
            setting left out leaves whatever is stored, so a model chosen in the dashboard survives a
            sync that says nothing about it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SyncAgentResult]
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
    body: SyncAgentRequest,
) -> ErrorResponse | SyncAgentResult | None:
    """Store an agent directory's instructions, skills, knowledge, simulations and settings

     Reads as "this is what the agent is", from a directory of agent.yaml, instructions.md, skills/,
    knowledge/ and simulations/. The hash is a fingerprint of that directory: a second call with the
    same hash does nothing, so a process that syncs on startup is cheap when nothing has changed.

    agent.yaml decides the models, the voice and the rest of a config, so an agent kept in a repository
    needs nothing written by hand. A setting it leaves out is left alone rather than blanked.

    knowledge/ is the whole of the knowledge base named after the agent, and simulations/ the whole of
    its simulations: a file taken out of the directory is taken out of the backend on the next sync.

    A directory is not the only thing that writes an agent: somebody may have changed one of the same
    settings in the dashboard since the last sync. Send `check_changes` and such a sync is refused with
    `unsynced_changes` instead of writing over them -- only when it really would write over them, so a
    directory that already holds what the dashboard says syncs without complaint. Read the changes from
    `GET /v1/agents/configs/{id}/changes`, let the person decide, and sync again with `base_change` to
    go ahead.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest): An agent directory as it is on disk. Everything after the
            simulations is what the directory's declaration decides rather than what it holds, and a
            setting left out leaves whatever is stored, so a model chosen in the dashboard survives a
            sync that says nothing about it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SyncAgentResult
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SyncAgentRequest,
) -> Response[ErrorResponse | SyncAgentResult]:
    """Store an agent directory's instructions, skills, knowledge, simulations and settings

     Reads as "this is what the agent is", from a directory of agent.yaml, instructions.md, skills/,
    knowledge/ and simulations/. The hash is a fingerprint of that directory: a second call with the
    same hash does nothing, so a process that syncs on startup is cheap when nothing has changed.

    agent.yaml decides the models, the voice and the rest of a config, so an agent kept in a repository
    needs nothing written by hand. A setting it leaves out is left alone rather than blanked.

    knowledge/ is the whole of the knowledge base named after the agent, and simulations/ the whole of
    its simulations: a file taken out of the directory is taken out of the backend on the next sync.

    A directory is not the only thing that writes an agent: somebody may have changed one of the same
    settings in the dashboard since the last sync. Send `check_changes` and such a sync is refused with
    `unsynced_changes` instead of writing over them -- only when it really would write over them, so a
    directory that already holds what the dashboard says syncs without complaint. Read the changes from
    `GET /v1/agents/configs/{id}/changes`, let the person decide, and sync again with `base_change` to
    go ahead.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest): An agent directory as it is on disk. Everything after the
            simulations is what the directory's declaration decides rather than what it holds, and a
            setting left out leaves whatever is stored, so a model chosen in the dashboard survives a
            sync that says nothing about it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SyncAgentResult]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: SyncAgentRequest,
) -> ErrorResponse | SyncAgentResult | None:
    """Store an agent directory's instructions, skills, knowledge, simulations and settings

     Reads as "this is what the agent is", from a directory of agent.yaml, instructions.md, skills/,
    knowledge/ and simulations/. The hash is a fingerprint of that directory: a second call with the
    same hash does nothing, so a process that syncs on startup is cheap when nothing has changed.

    agent.yaml decides the models, the voice and the rest of a config, so an agent kept in a repository
    needs nothing written by hand. A setting it leaves out is left alone rather than blanked.

    knowledge/ is the whole of the knowledge base named after the agent, and simulations/ the whole of
    its simulations: a file taken out of the directory is taken out of the backend on the next sync.

    A directory is not the only thing that writes an agent: somebody may have changed one of the same
    settings in the dashboard since the last sync. Send `check_changes` and such a sync is refused with
    `unsynced_changes` instead of writing over them -- only when it really would write over them, so a
    directory that already holds what the dashboard says syncs without complaint. Read the changes from
    `GET /v1/agents/configs/{id}/changes`, let the person decide, and sync again with `base_change` to
    go ahead.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest): An agent directory as it is on disk. Everything after the
            simulations is what the directory's declaration decides rather than what it holds, and a
            setting left out leaves whatever is stored, so a model chosen in the dashboard survives a
            sync that says nothing about it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SyncAgentResult
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
