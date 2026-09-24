from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.image_generation import ImageGeneration
from ...models.image_generation_request import ImageGenerationRequest
from ...types import Response


def _get_kwargs(
    *,
    body: ImageGenerationRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/image/generations",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | ImageGeneration | None:
    if response.status_code == 200:
        response_200 = ImageGeneration.from_dict(response.json())

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

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | ImageGeneration]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ImageGenerationRequest,
) -> Response[Error | ImageGeneration]:
    """Draw pictures from a prompt, and return them

     Routed like search: a target or a priority list picks the model, failover and billing work as they
    do everywhere else, and one request is one stat row counting its pictures. The pictures come back in
    the response as bytes, never as a link, and nothing is stored, so the id cannot be fetched again.
    The request is answered when the pictures are drawn, within 240 seconds; a caller that hangs up
    cancels the job at the provider. A generation that got as far as a provider answers 200 whether it
    drew or not: a failed one carries status failed, an error_code and the error, and costs nothing. A
    request that could not be routed at all, or that asks for something no model could draw, is a 400.
    A failed generation is asked of the next candidate only when the provider never accepted the job,
    and never after a safety filter refused it, since asking the next vendor is shopping for a laxer
    filter.

    Args:
        body (ImageGenerationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | ImageGeneration]
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
    body: ImageGenerationRequest,
) -> Error | ImageGeneration | None:
    """Draw pictures from a prompt, and return them

     Routed like search: a target or a priority list picks the model, failover and billing work as they
    do everywhere else, and one request is one stat row counting its pictures. The pictures come back in
    the response as bytes, never as a link, and nothing is stored, so the id cannot be fetched again.
    The request is answered when the pictures are drawn, within 240 seconds; a caller that hangs up
    cancels the job at the provider. A generation that got as far as a provider answers 200 whether it
    drew or not: a failed one carries status failed, an error_code and the error, and costs nothing. A
    request that could not be routed at all, or that asks for something no model could draw, is a 400.
    A failed generation is asked of the next candidate only when the provider never accepted the job,
    and never after a safety filter refused it, since asking the next vendor is shopping for a laxer
    filter.

    Args:
        body (ImageGenerationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | ImageGeneration
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ImageGenerationRequest,
) -> Response[Error | ImageGeneration]:
    """Draw pictures from a prompt, and return them

     Routed like search: a target or a priority list picks the model, failover and billing work as they
    do everywhere else, and one request is one stat row counting its pictures. The pictures come back in
    the response as bytes, never as a link, and nothing is stored, so the id cannot be fetched again.
    The request is answered when the pictures are drawn, within 240 seconds; a caller that hangs up
    cancels the job at the provider. A generation that got as far as a provider answers 200 whether it
    drew or not: a failed one carries status failed, an error_code and the error, and costs nothing. A
    request that could not be routed at all, or that asks for something no model could draw, is a 400.
    A failed generation is asked of the next candidate only when the provider never accepted the job,
    and never after a safety filter refused it, since asking the next vendor is shopping for a laxer
    filter.

    Args:
        body (ImageGenerationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | ImageGeneration]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ImageGenerationRequest,
) -> Error | ImageGeneration | None:
    """Draw pictures from a prompt, and return them

     Routed like search: a target or a priority list picks the model, failover and billing work as they
    do everywhere else, and one request is one stat row counting its pictures. The pictures come back in
    the response as bytes, never as a link, and nothing is stored, so the id cannot be fetched again.
    The request is answered when the pictures are drawn, within 240 seconds; a caller that hangs up
    cancels the job at the provider. A generation that got as far as a provider answers 200 whether it
    drew or not: a failed one carries status failed, an error_code and the error, and costs nothing. A
    request that could not be routed at all, or that asks for something no model could draw, is a 400.
    A failed generation is asked of the next candidate only when the provider never accepted the job,
    and never after a safety filter refused it, since asking the next vendor is shopping for a laxer
    filter.

    Args:
        body (ImageGenerationRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | ImageGeneration
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
