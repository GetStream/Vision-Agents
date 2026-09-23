import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.granularity import Granularity
from ...models.spend_bucket import SpendBucket
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    group_by: str | Unset = "modality",
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    limit: int | Unset = 6,
    tag: list[str] | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["group_by"] = group_by

    json_granularity: str | Unset = UNSET
    if not isinstance(granularity, Unset):
        json_granularity = granularity.value

    params["granularity"] = json_granularity

    json_from_ = from_.isoformat()
    params["from"] = json_from_

    json_to = to.isoformat()
    params["to"] = json_to

    params["limit"] = limit

    json_tag: list[str] | Unset = UNSET
    if not isinstance(tag, Unset):
        json_tag = tag

    params["tag"] = json_tag

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/stats/spend",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[SpendBucket] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = SpendBucket.from_dict(response_200_item_data)

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
) -> Response[Error | list[SpendBucket]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    group_by: str | Unset = "modality",
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    limit: int | Unset = 6,
    tag: list[str] | Unset = UNSET,
) -> Response[Error | list[SpendBucket]]:
    """What the calling customer spent, grouped

     Spend across every modality at once, which is what a bill is. group_by decides what the series are:
    "modality" for where the money went, or a cost label for what it was spent on.
    Only the biggest values keep a series of their own, because a label such as customer_id has as many
    values as the customer has customers. The rest are summed into "other", and spend carrying no such
    label at all into the empty value, so the rows still add up to the total.
    Reads the request rows rather than the rollups, so today's spend is there without a rollup having
    run.

    Args:
        group_by (str | Unset):  Default: 'modality'.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        limit (int | Unset):  Default: 6.
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[SpendBucket]]
    """

    kwargs = _get_kwargs(
        group_by=group_by,
        granularity=granularity,
        from_=from_,
        to=to,
        limit=limit,
        tag=tag,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    group_by: str | Unset = "modality",
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    limit: int | Unset = 6,
    tag: list[str] | Unset = UNSET,
) -> Error | list[SpendBucket] | None:
    """What the calling customer spent, grouped

     Spend across every modality at once, which is what a bill is. group_by decides what the series are:
    "modality" for where the money went, or a cost label for what it was spent on.
    Only the biggest values keep a series of their own, because a label such as customer_id has as many
    values as the customer has customers. The rest are summed into "other", and spend carrying no such
    label at all into the empty value, so the rows still add up to the total.
    Reads the request rows rather than the rollups, so today's spend is there without a rollup having
    run.

    Args:
        group_by (str | Unset):  Default: 'modality'.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        limit (int | Unset):  Default: 6.
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[SpendBucket]
    """

    return sync_detailed(
        client=client,
        group_by=group_by,
        granularity=granularity,
        from_=from_,
        to=to,
        limit=limit,
        tag=tag,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    group_by: str | Unset = "modality",
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    limit: int | Unset = 6,
    tag: list[str] | Unset = UNSET,
) -> Response[Error | list[SpendBucket]]:
    """What the calling customer spent, grouped

     Spend across every modality at once, which is what a bill is. group_by decides what the series are:
    "modality" for where the money went, or a cost label for what it was spent on.
    Only the biggest values keep a series of their own, because a label such as customer_id has as many
    values as the customer has customers. The rest are summed into "other", and spend carrying no such
    label at all into the empty value, so the rows still add up to the total.
    Reads the request rows rather than the rollups, so today's spend is there without a rollup having
    run.

    Args:
        group_by (str | Unset):  Default: 'modality'.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        limit (int | Unset):  Default: 6.
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[SpendBucket]]
    """

    kwargs = _get_kwargs(
        group_by=group_by,
        granularity=granularity,
        from_=from_,
        to=to,
        limit=limit,
        tag=tag,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    group_by: str | Unset = "modality",
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    limit: int | Unset = 6,
    tag: list[str] | Unset = UNSET,
) -> Error | list[SpendBucket] | None:
    """What the calling customer spent, grouped

     Spend across every modality at once, which is what a bill is. group_by decides what the series are:
    "modality" for where the money went, or a cost label for what it was spent on.
    Only the biggest values keep a series of their own, because a label such as customer_id has as many
    values as the customer has customers. The rest are summed into "other", and spend carrying no such
    label at all into the empty value, so the rows still add up to the total.
    Reads the request rows rather than the rollups, so today's spend is there without a rollup having
    run.

    Args:
        group_by (str | Unset):  Default: 'modality'.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        limit (int | Unset):  Default: 6.
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[SpendBucket]
    """

    return (
        await asyncio_detailed(
            client=client,
            group_by=group_by,
            granularity=granularity,
            from_=from_,
            to=to,
            limit=limit,
            tag=tag,
        )
    ).parsed
