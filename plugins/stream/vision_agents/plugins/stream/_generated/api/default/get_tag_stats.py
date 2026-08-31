import datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.granularity import Granularity
from ...models.modality import Modality
from ...models.tag_stats_bucket import TagStatsBucket
from ...types import UNSET, Response, Unset


def _get_kwargs(
    modality: Modality,
    *,
    key: str,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["key"] = key

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
        "url": "/v1/{modality}/stats/tags".format(
            modality=quote(str(modality), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[TagStatsBucket] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TagStatsBucket.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[TagStatsBucket]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
    key: str,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[Error | list[TagStatsBucket]]:
    """Aggregated usage broken down by the values of one cost label

     What drives the spend. Requests are labelled with whatever keys the customer chooses, so asking for
    key=project returns one row per project per bucket.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        key (str):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[TagStatsBucket]]
    """

    kwargs = _get_kwargs(
        modality=modality,
        key=key,
        granularity=granularity,
        from_=from_,
        to=to,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
    key: str,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Error | list[TagStatsBucket] | None:
    """Aggregated usage broken down by the values of one cost label

     What drives the spend. Requests are labelled with whatever keys the customer chooses, so asking for
    key=project returns one row per project per bucket.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        key (str):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[TagStatsBucket]
    """

    return sync_detailed(
        modality=modality,
        client=client,
        key=key,
        granularity=granularity,
        from_=from_,
        to=to,
    ).parsed


async def asyncio_detailed(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
    key: str,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Response[Error | list[TagStatsBucket]]:
    """Aggregated usage broken down by the values of one cost label

     What drives the spend. Requests are labelled with whatever keys the customer chooses, so asking for
    key=project returns one row per project per bucket.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        key (str):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[TagStatsBucket]]
    """

    kwargs = _get_kwargs(
        modality=modality,
        key=key,
        granularity=granularity,
        from_=from_,
        to=to,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
    key: str,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
) -> Error | list[TagStatsBucket] | None:
    """Aggregated usage broken down by the values of one cost label

     What drives the spend. Requests are labelled with whatever keys the customer chooses, so asking for
    key=project returns one row per project per bucket.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        key (str):
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[TagStatsBucket]
    """

    return (
        await asyncio_detailed(
            modality=modality,
            client=client,
            key=key,
            granularity=granularity,
            from_=from_,
            to=to,
        )
    ).parsed
