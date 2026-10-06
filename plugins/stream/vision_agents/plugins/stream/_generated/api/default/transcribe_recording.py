from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.transcription import Transcription
from ...models.transcription_request import TranscriptionRequest
from ...types import Response


def _get_kwargs(
    *,
    body: TranscriptionRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/stt/recordings",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | Transcription | None:
    if response.status_code == 202:
        response_202 = Transcription.from_dict(response.json())

        return response_202

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
) -> Response[ErrorResponse | Transcription]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TranscriptionRequest,
) -> Response[ErrorResponse | Transcription]:
    """Transcribe a recording, off the live path

     The non-realtime half of speech-to-text: a whole recording in, a whole transcript out. It is a job
    rather than a response because an hour of audio takes minutes to transcribe, so this returns
    immediately with an id to poll, or calls a callback when it is done.
    Routing works as it does everywhere else, except that the candidates are the providers registered as
    not realtime - the batch APIs, which are cheaper and more accurate than the same vendor's streaming
    model.

    Args:
        body (TranscriptionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Transcription]
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
    body: TranscriptionRequest,
) -> ErrorResponse | Transcription | None:
    """Transcribe a recording, off the live path

     The non-realtime half of speech-to-text: a whole recording in, a whole transcript out. It is a job
    rather than a response because an hour of audio takes minutes to transcribe, so this returns
    immediately with an id to poll, or calls a callback when it is done.
    Routing works as it does everywhere else, except that the candidates are the providers registered as
    not realtime - the batch APIs, which are cheaper and more accurate than the same vendor's streaming
    model.

    Args:
        body (TranscriptionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Transcription
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TranscriptionRequest,
) -> Response[ErrorResponse | Transcription]:
    """Transcribe a recording, off the live path

     The non-realtime half of speech-to-text: a whole recording in, a whole transcript out. It is a job
    rather than a response because an hour of audio takes minutes to transcribe, so this returns
    immediately with an id to poll, or calls a callback when it is done.
    Routing works as it does everywhere else, except that the candidates are the providers registered as
    not realtime - the batch APIs, which are cheaper and more accurate than the same vendor's streaming
    model.

    Args:
        body (TranscriptionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Transcription]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: TranscriptionRequest,
) -> ErrorResponse | Transcription | None:
    """Transcribe a recording, off the live path

     The non-realtime half of speech-to-text: a whole recording in, a whole transcript out. It is a job
    rather than a response because an hour of audio takes minutes to transcribe, so this returns
    immediately with an id to poll, or calls a callback when it is done.
    Routing works as it does everywhere else, except that the candidates are the providers registered as
    not realtime - the batch APIs, which are cheaper and more accurate than the same vendor's streaming
    model.

    Args:
        body (TranscriptionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Transcription
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
