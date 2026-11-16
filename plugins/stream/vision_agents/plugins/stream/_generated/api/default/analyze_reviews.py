from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.analyze_reviews_request import AnalyzeReviewsRequest
from ...models.error_response import ErrorResponse
from ...models.review_analysis import ReviewAnalysis
from ...types import Response


def _get_kwargs(
    *,
    body: AnalyzeReviewsRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/calls/reviews/analysis",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | ReviewAnalysis | None:
    if response.status_code == 200:
        response_200 = ReviewAnalysis.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | ReviewAnalysis]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: AnalyzeReviewsRequest,
) -> Response[ErrorResponse | ReviewAnalysis]:
    """What caused the calls the agents handled badly

     Reads the latest 100 calls in the window whose review scored them 3 or lower, and asks a model to
    group them by cause, each with the fix to make and the calls it explains.
    Costs one language model call, billed as any other, and keeps nothing: ask again for a fresh answer.

    Args:
        body (AnalyzeReviewsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ReviewAnalysis]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: AnalyzeReviewsRequest,
) -> ErrorResponse | ReviewAnalysis | None:
    """What caused the calls the agents handled badly

     Reads the latest 100 calls in the window whose review scored them 3 or lower, and asks a model to
    group them by cause, each with the fix to make and the calls it explains.
    Costs one language model call, billed as any other, and keeps nothing: ask again for a fresh answer.

    Args:
        body (AnalyzeReviewsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ReviewAnalysis
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: AnalyzeReviewsRequest,
) -> Response[ErrorResponse | ReviewAnalysis]:
    """What caused the calls the agents handled badly

     Reads the latest 100 calls in the window whose review scored them 3 or lower, and asks a model to
    group them by cause, each with the fix to make and the calls it explains.
    Costs one language model call, billed as any other, and keeps nothing: ask again for a fresh answer.

    Args:
        body (AnalyzeReviewsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ReviewAnalysis]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: AnalyzeReviewsRequest,
) -> ErrorResponse | ReviewAnalysis | None:
    """What caused the calls the agents handled badly

     Reads the latest 100 calls in the window whose review scored them 3 or lower, and asks a model to
    group them by cause, each with the fix to make and the calls it explains.
    Costs one language model call, billed as any other, and keeps nothing: ask again for a fresh answer.

    Args:
        body (AnalyzeReviewsRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ReviewAnalysis
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
