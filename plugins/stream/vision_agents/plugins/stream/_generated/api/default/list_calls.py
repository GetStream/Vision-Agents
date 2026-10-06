import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.call import Call
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    running: bool | Unset = False,
    limit: int | Unset = 50,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["running"] = running

    params["limit"] = limit

    params["agent_id"] = agent_id

    params["campaign_id"] = campaign_id

    json_from_: str | Unset = UNSET
    if not isinstance(from_, Unset):
        json_from_ = from_.isoformat()
    params["from"] = json_from_

    json_to: str | Unset = UNSET
    if not isinstance(to, Unset):
        json_to = to.isoformat()
    params["to"] = json_to

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/calls",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | list[Call] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Call.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | list[Call]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    running: bool | Unset = False,
    limit: int | Unset = 50,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
) -> Response[ErrorResponse | list[Call]]:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        running (bool | Unset):  Default: False.
        limit (int | Unset):  Default: 50.
        agent_id (str | Unset): Narrow to one agent.
        campaign_id (str | Unset): Narrow to the calls one campaign placed.
        from_ (datetime.datetime | Unset): Only calls that started at or after this, inclusive.
        to (datetime.datetime | Unset): Only calls that started before this, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | list[Call]]
    """

    kwargs = _get_kwargs(
        running=running,
        limit=limit,
        agent_id=agent_id,
        campaign_id=campaign_id,
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
    running: bool | Unset = False,
    limit: int | Unset = 50,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
) -> ErrorResponse | list[Call] | None:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        running (bool | Unset):  Default: False.
        limit (int | Unset):  Default: 50.
        agent_id (str | Unset): Narrow to one agent.
        campaign_id (str | Unset): Narrow to the calls one campaign placed.
        from_ (datetime.datetime | Unset): Only calls that started at or after this, inclusive.
        to (datetime.datetime | Unset): Only calls that started before this, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | list[Call]
    """

    return sync_detailed(
        client=client,
        running=running,
        limit=limit,
        agent_id=agent_id,
        campaign_id=campaign_id,
        from_=from_,
        to=to,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    running: bool | Unset = False,
    limit: int | Unset = 50,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
) -> Response[ErrorResponse | list[Call]]:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        running (bool | Unset):  Default: False.
        limit (int | Unset):  Default: 50.
        agent_id (str | Unset): Narrow to one agent.
        campaign_id (str | Unset): Narrow to the calls one campaign placed.
        from_ (datetime.datetime | Unset): Only calls that started at or after this, inclusive.
        to (datetime.datetime | Unset): Only calls that started before this, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | list[Call]]
    """

    kwargs = _get_kwargs(
        running=running,
        limit=limit,
        agent_id=agent_id,
        campaign_id=campaign_id,
        from_=from_,
        to=to,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    running: bool | Unset = False,
    limit: int | Unset = 50,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
) -> ErrorResponse | list[Call] | None:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        running (bool | Unset):  Default: False.
        limit (int | Unset):  Default: 50.
        agent_id (str | Unset): Narrow to one agent.
        campaign_id (str | Unset): Narrow to the calls one campaign placed.
        from_ (datetime.datetime | Unset): Only calls that started at or after this, inclusive.
        to (datetime.datetime | Unset): Only calls that started before this, exclusive.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | list[Call]
    """

    return (
        await asyncio_detailed(
            client=client,
            running=running,
            limit=limit,
            agent_id=agent_id,
            campaign_id=campaign_id,
            from_=from_,
            to=to,
        )
    ).parsed
