import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.tag_key_summary import TagKeySummary
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

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
        "url": "/v1/stats/tags/keys",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[TagKeySummary] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = TagKeySummary.from_dict(response_200_item_data)

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
) -> Response[Error | list[TagKeySummary]]:
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
    tag: list[str] | Unset = UNSET,
) -> Response[Error | list[TagKeySummary]]:
    """Which cost labels the calling customer's spend carries

     Cost labels are the customer's own, so nothing here knows in advance whether spend is broken down by
    product, by environment or by the end customer it was incurred for. This reports the keys in use and
    what each covers, so a reader can be shown the breakdown that means something rather than a list to
    guess from.
    A key every request carries with a single value -- environment: production and nothing else -- is
    context rather than a breakdown, and value_count says so.

    Args:
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[TagKeySummary]]
    """

    kwargs = _get_kwargs(
        from_=from_,
        to=to,
        tag=tag,
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
    tag: list[str] | Unset = UNSET,
) -> Error | list[TagKeySummary] | None:
    """Which cost labels the calling customer's spend carries

     Cost labels are the customer's own, so nothing here knows in advance whether spend is broken down by
    product, by environment or by the end customer it was incurred for. This reports the keys in use and
    what each covers, so a reader can be shown the breakdown that means something rather than a list to
    guess from.
    A key every request carries with a single value -- environment: production and nothing else -- is
    context rather than a breakdown, and value_count says so.

    Args:
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[TagKeySummary]
    """

    return sync_detailed(
        client=client,
        from_=from_,
        to=to,
        tag=tag,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
) -> Response[Error | list[TagKeySummary]]:
    """Which cost labels the calling customer's spend carries

     Cost labels are the customer's own, so nothing here knows in advance whether spend is broken down by
    product, by environment or by the end customer it was incurred for. This reports the keys in use and
    what each covers, so a reader can be shown the breakdown that means something rather than a list to
    guess from.
    A key every request carries with a single value -- environment: production and nothing else -- is
    context rather than a breakdown, and value_count says so.

    Args:
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[TagKeySummary]]
    """

    kwargs = _get_kwargs(
        from_=from_,
        to=to,
        tag=tag,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    from_: datetime.datetime,
    to: datetime.datetime,
    tag: list[str] | Unset = UNSET,
) -> Error | list[TagKeySummary] | None:
    """Which cost labels the calling customer's spend carries

     Cost labels are the customer's own, so nothing here knows in advance whether spend is broken down by
    product, by environment or by the end customer it was incurred for. This reports the keys in use and
    what each covers, so a reader can be shown the breakdown that means something rather than a list to
    guess from.
    A key every request carries with a single value -- environment: production and nothing else -- is
    context rather than a breakdown, and value_count says so.

    Args:
        from_ (datetime.datetime):
        to (datetime.datetime):
        tag (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[TagKeySummary]
    """

    return (
        await asyncio_detailed(
            client=client,
            from_=from_,
            to=to,
            tag=tag,
        )
    ).parsed
