from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_config import AgentConfig
from ...models.agent_config_request import AgentConfigRequest
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    *,
    body: AgentConfigRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/configs",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentConfig | ErrorResponse | None:
    if response.status_code == 201:
        response_201 = AgentConfig.from_dict(response.json())

        return response_201

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
) -> Response[AgentConfig | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: AgentConfigRequest,
) -> Response[AgentConfig | ErrorResponse]:
    """Store a named configuration a session can be created from

     A config holds what a caller would otherwise repeat on every call: the models, the voice, the
    instructions and which skills the subagent may be handed. What is about one conversation rather than
    the agent behind it, the call id above all, stays in the create-session request.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AgentConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentConfig | ErrorResponse]
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
    body: AgentConfigRequest,
) -> AgentConfig | ErrorResponse | None:
    """Store a named configuration a session can be created from

     A config holds what a caller would otherwise repeat on every call: the models, the voice, the
    instructions and which skills the subagent may be handed. What is about one conversation rather than
    the agent behind it, the call id above all, stays in the create-session request.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AgentConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentConfig | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: AgentConfigRequest,
) -> Response[AgentConfig | ErrorResponse]:
    """Store a named configuration a session can be created from

     A config holds what a caller would otherwise repeat on every call: the models, the voice, the
    instructions and which skills the subagent may be handed. What is about one conversation rather than
    the agent behind it, the call id above all, stays in the create-session request.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AgentConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentConfig | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: AgentConfigRequest,
) -> AgentConfig | ErrorResponse | None:
    """Store a named configuration a session can be created from

     A config holds what a caller would otherwise repeat on every call: the models, the voice, the
    instructions and which skills the subagent may be handed. What is about one conversation rather than
    the agent behind it, the call id above all, stays in the create-session request.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AgentConfigRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentConfig | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
