import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.granularity import Granularity
from ...models.turn_stats_bucket import TurnStatsBucket
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    agent_id: str | Unset = UNSET,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["agent_id"] = agent_id

    json_granularity: str | Unset = UNSET
    if not isinstance(granularity, Unset):
        json_granularity = granularity.value

    params["granularity"] = json_granularity

    json_from_ = from_.isoformat()
    params["from"] = json_from_

    json_to = to.isoformat()
    params["to"] = json_to

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/turns/stats",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[TurnStatsBucket] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TurnStatsBucket.from_dict(response_200_item_data)

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
) -> Response[Error | list[TurnStatsBucket]]:
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
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[Error | list[TurnStatsBucket]]:
    """Conversational latency for the calling customer

     One row per bucket and agent. A request row measures one provider call; a turn measures what the
    caller felt, from finishing a sentence to hearing the answer start, with the transcription, model
    and voice legs kept apart so a slow conversation can be attributed.

    Args:
        agent_id (str | Unset):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[TurnStatsBucket]]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        granularity=granularity,
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
    agent_id: str | Unset = UNSET,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Error | list[TurnStatsBucket] | None:
    """Conversational latency for the calling customer

     One row per bucket and agent. A request row measures one provider call; a turn measures what the
    caller felt, from finishing a sentence to hearing the answer start, with the transcription, model
    and voice legs kept apart so a slow conversation can be attributed.

    Args:
        agent_id (str | Unset):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[TurnStatsBucket]
    """

    return sync_detailed(
        client=client,
        agent_id=agent_id,
        granularity=granularity,
        from_=from_,
        to=to,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    agent_id: str | Unset = UNSET,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[Error | list[TurnStatsBucket]]:
    """Conversational latency for the calling customer

     One row per bucket and agent. A request row measures one provider call; a turn measures what the
    caller felt, from finishing a sentence to hearing the answer start, with the transcription, model
    and voice legs kept apart so a slow conversation can be attributed.

    Args:
        agent_id (str | Unset):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[TurnStatsBucket]]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        granularity=granularity,
        from_=from_,
        to=to,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    agent_id: str | Unset = UNSET,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Error | list[TurnStatsBucket] | None:
    """Conversational latency for the calling customer

     One row per bucket and agent. A request row measures one provider call; a turn measures what the
    caller felt, from finishing a sentence to hearing the answer start, with the transcription, model
    and voice legs kept apart so a slow conversation can be attributed.

    Args:
        agent_id (str | Unset):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[TurnStatsBucket]
    """

    return (
        await asyncio_detailed(
            client=client,
            agent_id=agent_id,
            granularity=granularity,
            from_=from_,
            to=to,
        )
    ).parsed
