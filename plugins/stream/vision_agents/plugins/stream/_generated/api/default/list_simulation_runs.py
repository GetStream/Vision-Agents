from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.list_simulation_runs_state import ListSimulationRunsState
from ...models.simulation_run import SimulationRun
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    simulation_id: str | Unset = UNSET,
    state: ListSimulationRunsState | Unset = UNSET,
    limit: int | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["simulation_id"] = simulation_id

    json_state: str | Unset = UNSET
    if not isinstance(state, Unset):
        json_state = state.value

    params["state"] = json_state

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/simulation-runs",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[SimulationRun] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = SimulationRun.from_dict(response_200_item_data)

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
) -> Response[Error | list[SimulationRun]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    simulation_id: str | Unset = UNSET,
    state: ListSimulationRunsState | Unset = UNSET,
    limit: int | Unset = UNSET,
) -> Response[Error | list[SimulationRun]]:
    """What the simulations have come to, newest first

     Without a simulation named this is the log of everything that has been run lately, which is the same
    question as what one simulation has come to, asked of all of them.

    Args:
        simulation_id (str | Unset):
        state (ListSimulationRunsState | Unset):
        limit (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[SimulationRun]]
    """

    kwargs = _get_kwargs(
        simulation_id=simulation_id,
        state=state,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    simulation_id: str | Unset = UNSET,
    state: ListSimulationRunsState | Unset = UNSET,
    limit: int | Unset = UNSET,
) -> Error | list[SimulationRun] | None:
    """What the simulations have come to, newest first

     Without a simulation named this is the log of everything that has been run lately, which is the same
    question as what one simulation has come to, asked of all of them.

    Args:
        simulation_id (str | Unset):
        state (ListSimulationRunsState | Unset):
        limit (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[SimulationRun]
    """

    return sync_detailed(
        client=client,
        simulation_id=simulation_id,
        state=state,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    simulation_id: str | Unset = UNSET,
    state: ListSimulationRunsState | Unset = UNSET,
    limit: int | Unset = UNSET,
) -> Response[Error | list[SimulationRun]]:
    """What the simulations have come to, newest first

     Without a simulation named this is the log of everything that has been run lately, which is the same
    question as what one simulation has come to, asked of all of them.

    Args:
        simulation_id (str | Unset):
        state (ListSimulationRunsState | Unset):
        limit (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[SimulationRun]]
    """

    kwargs = _get_kwargs(
        simulation_id=simulation_id,
        state=state,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    simulation_id: str | Unset = UNSET,
    state: ListSimulationRunsState | Unset = UNSET,
    limit: int | Unset = UNSET,
) -> Error | list[SimulationRun] | None:
    """What the simulations have come to, newest first

     Without a simulation named this is the log of everything that has been run lately, which is the same
    question as what one simulation has come to, asked of all of them.

    Args:
        simulation_id (str | Unset):
        state (ListSimulationRunsState | Unset):
        limit (int | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[SimulationRun]
    """

    return (
        await asyncio_detailed(
            client=client,
            simulation_id=simulation_id,
            state=state,
            limit=limit,
        )
    ).parsed
