from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.detection_request import DetectionRequest
from ...models.detection_result import DetectionResult
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    *,
    body: DetectionRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/detection",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> DetectionResult | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = DetectionResult.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[DetectionResult | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: DetectionRequest,
) -> Response[DetectionResult | ErrorResponse]:
    """Find objects in one image, and return their boxes

     Routed like image generation: a target or a priority list picks the model, failover and billing work
    as they do everywhere else, and one request is one stat row counting the images the vendor billed.
    Every box comes back as fractions of the image, whatever the vendor sent.
    detection-fast finds COCO's 80 classes quickly; detection-open finds whatever is named in classes;
    detection-faces finds faces with an embedding each. A confidence floor or embeddings narrow the
    candidates to the models that declared them, so they are honoured or the request is a 400.
    A detection that got as far as a provider answers 200 whether it found anything or not: a failed one
    carries status failed, an error_code and the error, and costs nothing. Nothing is stored, so the id
    cannot be fetched again.

    Args:
        body (DetectionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DetectionResult | ErrorResponse]
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
    body: DetectionRequest,
) -> DetectionResult | ErrorResponse | None:
    """Find objects in one image, and return their boxes

     Routed like image generation: a target or a priority list picks the model, failover and billing work
    as they do everywhere else, and one request is one stat row counting the images the vendor billed.
    Every box comes back as fractions of the image, whatever the vendor sent.
    detection-fast finds COCO's 80 classes quickly; detection-open finds whatever is named in classes;
    detection-faces finds faces with an embedding each. A confidence floor or embeddings narrow the
    candidates to the models that declared them, so they are honoured or the request is a 400.
    A detection that got as far as a provider answers 200 whether it found anything or not: a failed one
    carries status failed, an error_code and the error, and costs nothing. Nothing is stored, so the id
    cannot be fetched again.

    Args:
        body (DetectionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DetectionResult | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: DetectionRequest,
) -> Response[DetectionResult | ErrorResponse]:
    """Find objects in one image, and return their boxes

     Routed like image generation: a target or a priority list picks the model, failover and billing work
    as they do everywhere else, and one request is one stat row counting the images the vendor billed.
    Every box comes back as fractions of the image, whatever the vendor sent.
    detection-fast finds COCO's 80 classes quickly; detection-open finds whatever is named in classes;
    detection-faces finds faces with an embedding each. A confidence floor or embeddings narrow the
    candidates to the models that declared them, so they are honoured or the request is a 400.
    A detection that got as far as a provider answers 200 whether it found anything or not: a failed one
    carries status failed, an error_code and the error, and costs nothing. Nothing is stored, so the id
    cannot be fetched again.

    Args:
        body (DetectionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[DetectionResult | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: DetectionRequest,
) -> DetectionResult | ErrorResponse | None:
    """Find objects in one image, and return their boxes

     Routed like image generation: a target or a priority list picks the model, failover and billing work
    as they do everywhere else, and one request is one stat row counting the images the vendor billed.
    Every box comes back as fractions of the image, whatever the vendor sent.
    detection-fast finds COCO's 80 classes quickly; detection-open finds whatever is named in classes;
    detection-faces finds faces with an embedding each. A confidence floor or embeddings narrow the
    candidates to the models that declared them, so they are honoured or the request is a 400.
    A detection that got as far as a provider answers 200 whether it found anything or not: a failed one
    carries status failed, an error_code and the error, and costs nothing. Nothing is stored, so the id
    cannot be fetched again.

    Args:
        body (DetectionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        DetectionResult | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
