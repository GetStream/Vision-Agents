import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.call import Call
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    running: bool | Unset = False,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    limit: int | Unset = 50,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["agent_id"] = agent_id

    params["campaign_id"] = campaign_id

    params["running"] = running

    json_from_: str | Unset = UNSET
    if not isinstance(from_, Unset):
        json_from_ = from_.isoformat()
    params["from"] = json_from_

    json_to: str | Unset = UNSET
    if not isinstance(to, Unset):
        json_to = to.isoformat()
    params["to"] = json_to

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/calls",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[Call] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Call.from_dict(response_200_item_data)

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
) -> Response[Error | list[Call]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    running: bool | Unset = False,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    limit: int | Unset = 50,
) -> Response[Error | list[Call]]:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        agent_id (str | Unset):
        campaign_id (str | Unset):
        running (bool | Unset):  Default: False.
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        limit (int | Unset):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Call]]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        campaign_id=campaign_id,
        running=running,
        from_=from_,
        to=to,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    running: bool | Unset = False,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    limit: int | Unset = 50,
) -> Error | list[Call] | None:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        agent_id (str | Unset):
        campaign_id (str | Unset):
        running (bool | Unset):  Default: False.
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        limit (int | Unset):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Call]
    """

    return sync_detailed(
        client=client,
        agent_id=agent_id,
        campaign_id=campaign_id,
        running=running,
        from_=from_,
        to=to,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    running: bool | Unset = False,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    limit: int | Unset = 50,
) -> Response[Error | list[Call]]:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        agent_id (str | Unset):
        campaign_id (str | Unset):
        running (bool | Unset):  Default: False.
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        limit (int | Unset):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Call]]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        campaign_id=campaign_id,
        running=running,
        from_=from_,
        to=to,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    agent_id: str | Unset = UNSET,
    campaign_id: str | Unset = UNSET,
    running: bool | Unset = False,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    limit: int | Unset = 50,
) -> Error | list[Call] | None:
    """The calls the calling customer has run

     A session lives in memory and is gone when the process is, so a call is recorded as it starts and
    again as it ends. This is what answers what happened yesterday, and what is happening now after a
    restart.

    Args:
        agent_id (str | Unset):
        campaign_id (str | Unset):
        running (bool | Unset):  Default: False.
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        limit (int | Unset):  Default: 50.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Call]
    """

    return (
        await asyncio_detailed(
            client=client,
            agent_id=agent_id,
            campaign_id=campaign_id,
            running=running,
            from_=from_,
            to=to,
            limit=limit,
        )
    ).parsed
