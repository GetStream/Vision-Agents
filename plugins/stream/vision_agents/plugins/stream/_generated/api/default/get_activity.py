import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.activity_bucket import ActivityBucket
from ...models.activity_granularity import ActivityGranularity
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    granularity: ActivityGranularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

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
        "url": "/v1/stats/activity",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[ActivityBucket] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = ActivityBucket.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[Error | list[ActivityBucket]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    granularity: ActivityGranularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[Error | list[ActivityBucket]]:
    """Who used the calling customer's agents, and how much

     Sessions opened, responses produced and calls held, counted per bucket, alongside how many distinct
    people were behind them.
    Distinct users are counted rather than summed, which is why the granularity here is days or months
    rather than the hours the spend paths take: a month's active users are the people who came back, not
    the sum of its days, so a month has to be asked for as a month.

    Args:
        granularity (ActivityGranularity | Unset): Separate from Granularity, and coarser, because
            distinct users cannot be summed: a month of them is who came back rather than the sum of
            its days.
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[ActivityBucket]]
    """

    kwargs = _get_kwargs(
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
    granularity: ActivityGranularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Error | list[ActivityBucket] | None:
    """Who used the calling customer's agents, and how much

     Sessions opened, responses produced and calls held, counted per bucket, alongside how many distinct
    people were behind them.
    Distinct users are counted rather than summed, which is why the granularity here is days or months
    rather than the hours the spend paths take: a month's active users are the people who came back, not
    the sum of its days, so a month has to be asked for as a month.

    Args:
        granularity (ActivityGranularity | Unset): Separate from Granularity, and coarser, because
            distinct users cannot be summed: a month of them is who came back rather than the sum of
            its days.
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[ActivityBucket]
    """

    return sync_detailed(
        client=client,
        granularity=granularity,
        from_=from_,
        to=to,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    granularity: ActivityGranularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[Error | list[ActivityBucket]]:
    """Who used the calling customer's agents, and how much

     Sessions opened, responses produced and calls held, counted per bucket, alongside how many distinct
    people were behind them.
    Distinct users are counted rather than summed, which is why the granularity here is days or months
    rather than the hours the spend paths take: a month's active users are the people who came back, not
    the sum of its days, so a month has to be asked for as a month.

    Args:
        granularity (ActivityGranularity | Unset): Separate from Granularity, and coarser, because
            distinct users cannot be summed: a month of them is who came back rather than the sum of
            its days.
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[ActivityBucket]]
    """

    kwargs = _get_kwargs(
        granularity=granularity,
        from_=from_,
        to=to,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    granularity: ActivityGranularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Error | list[ActivityBucket] | None:
    """Who used the calling customer's agents, and how much

     Sessions opened, responses produced and calls held, counted per bucket, alongside how many distinct
    people were behind them.
    Distinct users are counted rather than summed, which is why the granularity here is days or months
    rather than the hours the spend paths take: a month's active users are the people who came back, not
    the sum of its days, so a month has to be asked for as a month.

    Args:
        granularity (ActivityGranularity | Unset): Separate from Granularity, and coarser, because
            distinct users cannot be summed: a month of them is who came back rather than the sum of
            its days.
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[ActivityBucket]
    """

    return (
        await asyncio_detailed(
            client=client,
            granularity=granularity,
            from_=from_,
            to=to,
        )
    ).parsed
