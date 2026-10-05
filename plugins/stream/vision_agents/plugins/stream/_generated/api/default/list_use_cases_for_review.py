from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.review_queue import ReviewQueue
from ...models.use_case_status import UseCaseStatus
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    status: UseCaseStatus | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_status: str | Unset = UNSET
    if not isinstance(status, Unset):
        json_status = status.value

    params["status"] = json_status

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/ops/use-cases",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | ReviewQueue | None:
    if response.status_code == 200:
        response_200 = ReviewQueue.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = Error.from_dict(response.json())

        return response_409

    if response.status_code == 500:
        response_500 = Error.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | ReviewQueue]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    status: UseCaseStatus | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[Error | ReviewQueue]:
    """List use cases waiting on Stream

     Every app's use cases in one status, longest waiting first, with the profile each was submitted on.

    Stream staff only: it needs the ops key.

    Args:
        status (UseCaseStatus | Unset): Where a use case stands. draft, changes_requested and
            vendor_rejected can be edited and submitted; submitted waits on Stream's review;
            vendor_pending on the vendor's; approved numbers may send. rejected is final.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same status.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | ReviewQueue]
    """

    kwargs = _get_kwargs(
        status=status,
        limit=limit,
        cursor=cursor,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    status: UseCaseStatus | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Error | ReviewQueue | None:
    """List use cases waiting on Stream

     Every app's use cases in one status, longest waiting first, with the profile each was submitted on.

    Stream staff only: it needs the ops key.

    Args:
        status (UseCaseStatus | Unset): Where a use case stands. draft, changes_requested and
            vendor_rejected can be edited and submitted; submitted waits on Stream's review;
            vendor_pending on the vendor's; approved numbers may send. rejected is final.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same status.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | ReviewQueue
    """

    return sync_detailed(
        client=client,
        status=status,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    status: UseCaseStatus | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[Error | ReviewQueue]:
    """List use cases waiting on Stream

     Every app's use cases in one status, longest waiting first, with the profile each was submitted on.

    Stream staff only: it needs the ops key.

    Args:
        status (UseCaseStatus | Unset): Where a use case stands. draft, changes_requested and
            vendor_rejected can be edited and submitted; submitted waits on Stream's review;
            vendor_pending on the vendor's; approved numbers may send. rejected is final.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same status.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | ReviewQueue]
    """

    kwargs = _get_kwargs(
        status=status,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    status: UseCaseStatus | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Error | ReviewQueue | None:
    """List use cases waiting on Stream

     Every app's use cases in one status, longest waiting first, with the profile each was submitted on.

    Stream staff only: it needs the ops key.

    Args:
        status (UseCaseStatus | Unset): Where a use case stands. draft, changes_requested and
            vendor_rejected can be edited and submitted; submitted waits on Stream's review;
            vendor_pending on the vendor's; approved numbers may send. rejected is final.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same status.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | ReviewQueue
    """

    return (
        await asyncio_detailed(
            client=client,
            status=status,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
