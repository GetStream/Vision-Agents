from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
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
) -> Error | SyncAgentResult | None:
    if response.status_code == 200:
        response_200 = SyncAgentResult.from_dict(response.json())

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

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | SyncAgentResult]:
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
) -> Response[Error | SyncAgentResult]:
    """Store an agent directory's instructions, skills and knowledge

     Reads as "this is what the agent is", from a directory of instructions.md, skills/ and knowledge/.
    The hash is a fingerprint of that directory: a second call with the same hash does nothing, so a
    process that syncs on startup is cheap when nothing has changed.
    Models, voice and the rest of a config are left alone. This path only writes what a directory can
    hold.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | SyncAgentResult]
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
) -> Error | SyncAgentResult | None:
    """Store an agent directory's instructions, skills and knowledge

     Reads as "this is what the agent is", from a directory of instructions.md, skills/ and knowledge/.
    The hash is a fingerprint of that directory: a second call with the same hash does nothing, so a
    process that syncs on startup is cheap when nothing has changed.
    Models, voice and the rest of a config are left alone. This path only writes what a directory can
    hold.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | SyncAgentResult
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SyncAgentRequest,
) -> Response[Error | SyncAgentResult]:
    """Store an agent directory's instructions, skills and knowledge

     Reads as "this is what the agent is", from a directory of instructions.md, skills/ and knowledge/.
    The hash is a fingerprint of that directory: a second call with the same hash does nothing, so a
    process that syncs on startup is cheap when nothing has changed.
    Models, voice and the rest of a config are left alone. This path only writes what a directory can
    hold.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | SyncAgentResult]
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
) -> Error | SyncAgentResult | None:
    """Store an agent directory's instructions, skills and knowledge

     Reads as "this is what the agent is", from a directory of instructions.md, skills/ and knowledge/.
    The hash is a fingerprint of that directory: a second call with the same hash does nothing, so a
    process that syncs on startup is cheap when nothing has changed.
    Models, voice and the rest of a config are left alone. This path only writes what a directory can
    hold.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | SyncAgentResult
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
