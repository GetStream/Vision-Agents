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
from ...models.stats_bucket import StatsBucket
from ...types import UNSET, Response, Unset


def _get_kwargs(
    modality: Modality,
    *,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
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

    json_tag: list[str] | Unset = UNSET
    if not isinstance(tag, Unset):
        json_tag = tag

    params["tag"] = json_tag

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/{modality}/stats".format(
            modality=quote(str(modality), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[StatsBucket] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = StatsBucket.from_dict(response_200_item_data)

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
) -> Response[Error | list[StatsBucket]]:
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
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
) -> Response[Error | list[StatsBucket]]:
    """Aggregated usage for the calling customer

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[StatsBucket]]
    """

    kwargs = _get_kwargs(
        modality=modality,
        granularity=granularity,
        from_=from_,
        to=to,
        tag=tag,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
) -> Error | list[StatsBucket] | None:
    """Aggregated usage for the calling customer

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[StatsBucket]
    """

    return sync_detailed(
        modality=modality,
        client=client,
        granularity=granularity,
        from_=from_,
        to=to,
        tag=tag,
    ).parsed


async def asyncio_detailed(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
) -> Response[Error | list[StatsBucket]]:
    """Aggregated usage for the calling customer

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[StatsBucket]]
    """

    kwargs = _get_kwargs(
        modality=modality,
        granularity=granularity,
        from_=from_,
        to=to,
        tag=tag,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
    granularity: Granularity | Unset = UNSET,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
) -> Error | list[StatsBucket] | None:
    """Aggregated usage for the calling customer

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.
        granularity (Granularity | Unset):
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[StatsBucket]
    """

    return (
        await asyncio_detailed(
            modality=modality,
            client=client,
            granularity=granularity,
            from_=from_,
            to=to,
            tag=tag,
        )
    ).parsed
