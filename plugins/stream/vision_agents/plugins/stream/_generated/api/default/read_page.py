from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.ocr_request import OcrRequest
from ...models.ocr_result import OcrResult
from ...types import Response


def _get_kwargs(
    *,
    body: OcrRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/ocr",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | OcrResult | None:
    if response.status_code == 200:
        response_200 = OcrResult.from_dict(response.json())

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

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

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
) -> Response[ErrorResponse | OcrResult]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: OcrRequest,
) -> Response[ErrorResponse | OcrResult]:
    """Read the text of one page, and return it as Markdown

     Routed like detection: a target or a priority list picks the model, failover and billing work as
    they do everywhere else, and one page is one stat row. ocr-default asks Mistral OCR and fails over
    to vision language models that transcribe the page, and to nothing that cannot read one; ocr-vision
    asks only those. Mistral OCR is billed per page and a vision model by the tokens it took.
    A candidate that fails, or takes longer than 45 seconds, hands the page to the next one. A reading
    that got as far as a provider answers 200 whether it read anything or not: a failed one carries
    status failed, an error_code and the error, and costs nothing. A page with no text on it is
    completed with empty markdown. Nothing is stored, so the id cannot be fetched again.

    Args:
        body (OcrRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | OcrResult]
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
    body: OcrRequest,
) -> ErrorResponse | OcrResult | None:
    """Read the text of one page, and return it as Markdown

     Routed like detection: a target or a priority list picks the model, failover and billing work as
    they do everywhere else, and one page is one stat row. ocr-default asks Mistral OCR and fails over
    to vision language models that transcribe the page, and to nothing that cannot read one; ocr-vision
    asks only those. Mistral OCR is billed per page and a vision model by the tokens it took.
    A candidate that fails, or takes longer than 45 seconds, hands the page to the next one. A reading
    that got as far as a provider answers 200 whether it read anything or not: a failed one carries
    status failed, an error_code and the error, and costs nothing. A page with no text on it is
    completed with empty markdown. Nothing is stored, so the id cannot be fetched again.

    Args:
        body (OcrRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | OcrResult
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: OcrRequest,
) -> Response[ErrorResponse | OcrResult]:
    """Read the text of one page, and return it as Markdown

     Routed like detection: a target or a priority list picks the model, failover and billing work as
    they do everywhere else, and one page is one stat row. ocr-default asks Mistral OCR and fails over
    to vision language models that transcribe the page, and to nothing that cannot read one; ocr-vision
    asks only those. Mistral OCR is billed per page and a vision model by the tokens it took.
    A candidate that fails, or takes longer than 45 seconds, hands the page to the next one. A reading
    that got as far as a provider answers 200 whether it read anything or not: a failed one carries
    status failed, an error_code and the error, and costs nothing. A page with no text on it is
    completed with empty markdown. Nothing is stored, so the id cannot be fetched again.

    Args:
        body (OcrRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | OcrResult]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: OcrRequest,
) -> ErrorResponse | OcrResult | None:
    """Read the text of one page, and return it as Markdown

     Routed like detection: a target or a priority list picks the model, failover and billing work as
    they do everywhere else, and one page is one stat row. ocr-default asks Mistral OCR and fails over
    to vision language models that transcribe the page, and to nothing that cannot read one; ocr-vision
    asks only those. Mistral OCR is billed per page and a vision model by the tokens it took.
    A candidate that fails, or takes longer than 45 seconds, hands the page to the next one. A reading
    that got as far as a provider answers 200 whether it read anything or not: a failed one carries
    status failed, an error_code and the error, and costs nothing. A page with no text on it is
    completed with empty markdown. Nothing is stored, so the id cannot be fetched again.

    Args:
        body (OcrRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | OcrResult
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
