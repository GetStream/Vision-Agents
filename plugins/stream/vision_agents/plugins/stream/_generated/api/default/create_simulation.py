from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.simulation import Simulation
from ...models.simulation_request import SimulationRequest
from ...types import Response


def _get_kwargs(
    *,
    body: SimulationRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/simulations",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | Simulation | None:
    if response.status_code == 201:
        response_201 = Simulation.from_dict(response.json())

        return response_201

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
) -> Response[Error | Simulation]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SimulationRequest,
) -> Response[Error | Simulation]:
    """Write down a conversation to have with an agent

     A simulation is a scenario to put an agent through and something that has to be true at the end of
    it. The scenario is a brief rather than a script: it is given to a model that plays the caller,
    reads what the agent says back and decides what to say next, so a scenario can say to change your
    mind once the order is handled. Nothing is run until it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SimulationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Simulation]
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
    body: SimulationRequest,
) -> Error | Simulation | None:
    """Write down a conversation to have with an agent

     A simulation is a scenario to put an agent through and something that has to be true at the end of
    it. The scenario is a brief rather than a script: it is given to a model that plays the caller,
    reads what the agent says back and decides what to say next, so a scenario can say to change your
    mind once the order is handled. Nothing is run until it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SimulationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Simulation
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SimulationRequest,
) -> Response[Error | Simulation]:
    """Write down a conversation to have with an agent

     A simulation is a scenario to put an agent through and something that has to be true at the end of
    it. The scenario is a brief rather than a script: it is given to a model that plays the caller,
    reads what the agent says back and decides what to say next, so a scenario can say to change your
    mind once the order is handled. Nothing is run until it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SimulationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Simulation]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: SimulationRequest,
) -> Error | Simulation | None:
    """Write down a conversation to have with an agent

     A simulation is a scenario to put an agent through and something that has to be true at the end of
    it. The scenario is a brief rather than a script: it is given to a model that plays the caller,
    reads what the agent says back and decides what to say next, so a scenario can say to change your
    mind once the order is handled. Nothing is run until it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SimulationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Simulation
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
