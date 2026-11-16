import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_stats import AgentStats
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response


def _get_kwargs(
    *,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_from_ = from_.isoformat()
    params["from"] = json_from_

    json_to = to.isoformat()
    params["to"] = json_to

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/stats/agents",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentStats | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = AgentStats.from_dict(response.json())

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
) -> Response[AgentStats | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[AgentStats | ErrorResponse]:
    """How the calling customer's agents ran, across all of them

     The most agents that were running at once, and which tools they called: how often, how much each
    call put in front of the model and how long it took.
    One figure for the whole window rather than a row per bucket, because neither a peak nor a
    percentile can be summed out of smaller ones.

    Args:
        from_ (datetime.datetime): Start of the window, inclusive.
        to (datetime.datetime): End of the window, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentStats | ErrorResponse]
    """

    kwargs = _get_kwargs(
        from_=from_,
        to=to,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> AgentStats | ErrorResponse | None:
    """How the calling customer's agents ran, across all of them

     The most agents that were running at once, and which tools they called: how often, how much each
    call put in front of the model and how long it took.
    One figure for the whole window rather than a row per bucket, because neither a peak nor a
    percentile can be summed out of smaller ones.

    Args:
        from_ (datetime.datetime): Start of the window, inclusive.
        to (datetime.datetime): End of the window, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentStats | ErrorResponse
    """

    return sync_detailed(
        client=client,
        from_=from_,
        to=to,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[AgentStats | ErrorResponse]:
    """How the calling customer's agents ran, across all of them

     The most agents that were running at once, and which tools they called: how often, how much each
    call put in front of the model and how long it took.
    One figure for the whole window rather than a row per bucket, because neither a peak nor a
    percentile can be summed out of smaller ones.

    Args:
        from_ (datetime.datetime): Start of the window, inclusive.
        to (datetime.datetime): End of the window, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentStats | ErrorResponse]
    """

    kwargs = _get_kwargs(
        from_=from_,
        to=to,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> AgentStats | ErrorResponse | None:
    """How the calling customer's agents ran, across all of them

     The most agents that were running at once, and which tools they called: how often, how much each
    call put in front of the model and how long it took.
    One figure for the whole window rather than a row per bucket, because neither a peak nor a
    percentile can be summed out of smaller ones.

    Args:
        from_ (datetime.datetime): Start of the window, inclusive.
        to (datetime.datetime): End of the window, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentStats | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            from_=from_,
            to=to,
        )
    ).parsed
