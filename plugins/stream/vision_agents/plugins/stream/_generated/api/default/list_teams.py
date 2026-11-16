from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.team import Team
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    app_id: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["app_id"] = app_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/teams",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | list[Team] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Team.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | list[Team]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    app_id: str | Unset = UNSET,
) -> Response[ErrorResponse | list[Team]]:
    """List the organization's teams

     Every team a spend limit names or this month's work carried, by name, with the apps that carried it
    this month, what it spent in each current window and its limits, each with what it can spend once
    its parents are counted. A team has no members: a user's token names its team. With app_id, only
    what happened within that app. Spend is the usage service's, or with none the request rows summed.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        app_id (str | Unset): Only what happened within this app: each team's spend and limits
            there, and only the teams it saw.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | list[Team]]
    """

    kwargs = _get_kwargs(
        app_id=app_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    app_id: str | Unset = UNSET,
) -> ErrorResponse | list[Team] | None:
    """List the organization's teams

     Every team a spend limit names or this month's work carried, by name, with the apps that carried it
    this month, what it spent in each current window and its limits, each with what it can spend once
    its parents are counted. A team has no members: a user's token names its team. With app_id, only
    what happened within that app. Spend is the usage service's, or with none the request rows summed.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        app_id (str | Unset): Only what happened within this app: each team's spend and limits
            there, and only the teams it saw.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | list[Team]
    """

    return sync_detailed(
        client=client,
        app_id=app_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    app_id: str | Unset = UNSET,
) -> Response[ErrorResponse | list[Team]]:
    """List the organization's teams

     Every team a spend limit names or this month's work carried, by name, with the apps that carried it
    this month, what it spent in each current window and its limits, each with what it can spend once
    its parents are counted. A team has no members: a user's token names its team. With app_id, only
    what happened within that app. Spend is the usage service's, or with none the request rows summed.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        app_id (str | Unset): Only what happened within this app: each team's spend and limits
            there, and only the teams it saw.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | list[Team]]
    """

    kwargs = _get_kwargs(
        app_id=app_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    app_id: str | Unset = UNSET,
) -> ErrorResponse | list[Team] | None:
    """List the organization's teams

     Every team a spend limit names or this month's work carried, by name, with the apps that carried it
    this month, what it spent in each current window and its limits, each with what it can spend once
    its parents are counted. A team has no members: a user's token names its team. With app_id, only
    what happened within that app. Spend is the usage service's, or with none the request rows summed.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        app_id (str | Unset): Only what happened within this app: each team's spend and limits
            there, and only the teams it saw.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | list[Team]
    """

    return (
        await asyncio_detailed(
            client=client,
            app_id=app_id,
        )
    ).parsed
