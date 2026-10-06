from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.review_use_case_request import ReviewUseCaseRequest
from ...models.use_case_for_review import UseCaseForReview
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    body: ReviewUseCaseRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/ops/use-cases/{id}/review".format(
            id=quote(str(id), safe=""),
        ),
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | UseCaseForReview | None:
    if response.status_code == 200:
        response_200 = UseCaseForReview.from_dict(response.json())

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
) -> Response[Error | UseCaseForReview]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: AuthenticatedClient,
    body: ReviewUseCaseRequest | Unset = UNSET,
) -> Response[Error | UseCaseForReview]:
    """Review a submitted use case

     Approves, rejects or hands back a submitted use case. Approving registers the brand and the campaign
    with the vendor, which approves it in turn.

    Stream staff only: it needs the ops key.

    Args:
        id (str):
        body (ReviewUseCaseRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | UseCaseForReview]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient,
    body: ReviewUseCaseRequest | Unset = UNSET,
) -> Error | UseCaseForReview | None:
    """Review a submitted use case

     Approves, rejects or hands back a submitted use case. Approving registers the brand and the campaign
    with the vendor, which approves it in turn.

    Stream staff only: it needs the ops key.

    Args:
        id (str):
        body (ReviewUseCaseRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | UseCaseForReview
    """

    return sync_detailed(
        id=id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient,
    body: ReviewUseCaseRequest | Unset = UNSET,
) -> Response[Error | UseCaseForReview]:
    """Review a submitted use case

     Approves, rejects or hands back a submitted use case. Approving registers the brand and the campaign
    with the vendor, which approves it in turn.

    Stream staff only: it needs the ops key.

    Args:
        id (str):
        body (ReviewUseCaseRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | UseCaseForReview]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient,
    body: ReviewUseCaseRequest | Unset = UNSET,
) -> Error | UseCaseForReview | None:
    """Review a submitted use case

     Approves, rejects or hands back a submitted use case. Approving registers the brand and the campaign
    with the vendor, which approves it in turn.

    Stream staff only: it needs the ops key.

    Args:
        id (str):
        body (ReviewUseCaseRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | UseCaseForReview
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
