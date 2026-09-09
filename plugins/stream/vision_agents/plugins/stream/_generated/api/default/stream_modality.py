from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.modality import Modality
from ...types import Response


def _get_kwargs(
    modality: Modality,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/{modality}/stream".format(
            modality=quote(str(modality), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | Error | None:
    if response.status_code == 101:
        response_101 = cast(Any, None)
        return response_101

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
) -> Response[Any | Error]:
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
) -> Response[Any | Error]:
    """Route one modality over a socket, for a pipeline running elsewhere

     A WebSocket, which OpenAPI cannot describe past the upgrade. This is the routing the agent does,
    offered a piece at a time: a caller running its own pipeline sends audio or text and gets
    transcripts, audio or completions back, and the request is failed over and billed exactly as it
    would be inside a session.
    Every socket opens with a `start` frame. It names either a `config_id`, a stored router config to
    take the options from, or the options outright; naming both overrides that config field by field.
    What it may carry is the modality's own option block - `SttOptions` for speech-to-text, `TtsOptions`
    for a voice, `LlmOptions` for a model - plus `agent_id` and `call_id` to attribute the work to a
    conversation and `tags` to bill it.
    Speech-to-text then takes binary PCM at the `sample_rate` the start frame named, 16 kHz mono by
    default, and returns `transcript` frames. Text-to-speech takes `speak` frames and returns binary
    audio with `synthesis_complete` between utterances: each audio frame opens with a little-endian
    header of a uint32 sample rate, a uint16 channel count and two reserved bytes, followed by PCM16
    samples. Language models take `respond` frames, each naming an `id` and anything from `LlmOptions`
    for that one response along with its `tools`, and return `delta`, `reasoning_delta` and one
    `complete` per response; a `complete` reports the `status` the response ended in, what it cost in
    tokens and how long the caller waited for the first of them. An `interrupt` frame naming
    `response_ids` abandons responses still being generated, which still settle and are still billed for
    what they produced before being cut off. All three report failures as `error` frames and end with
    `closed`.
    Search is answered at `/v1/search` rather than here: one question and its answer need no socket held
    open between them. Memory and phone are recorded rather than routed, so they are not served either.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error]
    """

    kwargs = _get_kwargs(
        modality=modality,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
) -> Any | Error | None:
    """Route one modality over a socket, for a pipeline running elsewhere

     A WebSocket, which OpenAPI cannot describe past the upgrade. This is the routing the agent does,
    offered a piece at a time: a caller running its own pipeline sends audio or text and gets
    transcripts, audio or completions back, and the request is failed over and billed exactly as it
    would be inside a session.
    Every socket opens with a `start` frame. It names either a `config_id`, a stored router config to
    take the options from, or the options outright; naming both overrides that config field by field.
    What it may carry is the modality's own option block - `SttOptions` for speech-to-text, `TtsOptions`
    for a voice, `LlmOptions` for a model - plus `agent_id` and `call_id` to attribute the work to a
    conversation and `tags` to bill it.
    Speech-to-text then takes binary PCM at the `sample_rate` the start frame named, 16 kHz mono by
    default, and returns `transcript` frames. Text-to-speech takes `speak` frames and returns binary
    audio with `synthesis_complete` between utterances: each audio frame opens with a little-endian
    header of a uint32 sample rate, a uint16 channel count and two reserved bytes, followed by PCM16
    samples. Language models take `respond` frames, each naming an `id` and anything from `LlmOptions`
    for that one response along with its `tools`, and return `delta`, `reasoning_delta` and one
    `complete` per response; a `complete` reports the `status` the response ended in, what it cost in
    tokens and how long the caller waited for the first of them. An `interrupt` frame naming
    `response_ids` abandons responses still being generated, which still settle and are still billed for
    what they produced before being cut off. All three report failures as `error` frames and end with
    `closed`.
    Search is answered at `/v1/search` rather than here: one question and its answer need no socket held
    open between them. Memory and phone are recorded rather than routed, so they are not served either.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error
    """

    return sync_detailed(
        modality=modality,
        client=client,
    ).parsed


async def asyncio_detailed(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | Error]:
    """Route one modality over a socket, for a pipeline running elsewhere

     A WebSocket, which OpenAPI cannot describe past the upgrade. This is the routing the agent does,
    offered a piece at a time: a caller running its own pipeline sends audio or text and gets
    transcripts, audio or completions back, and the request is failed over and billed exactly as it
    would be inside a session.
    Every socket opens with a `start` frame. It names either a `config_id`, a stored router config to
    take the options from, or the options outright; naming both overrides that config field by field.
    What it may carry is the modality's own option block - `SttOptions` for speech-to-text, `TtsOptions`
    for a voice, `LlmOptions` for a model - plus `agent_id` and `call_id` to attribute the work to a
    conversation and `tags` to bill it.
    Speech-to-text then takes binary PCM at the `sample_rate` the start frame named, 16 kHz mono by
    default, and returns `transcript` frames. Text-to-speech takes `speak` frames and returns binary
    audio with `synthesis_complete` between utterances: each audio frame opens with a little-endian
    header of a uint32 sample rate, a uint16 channel count and two reserved bytes, followed by PCM16
    samples. Language models take `respond` frames, each naming an `id` and anything from `LlmOptions`
    for that one response along with its `tools`, and return `delta`, `reasoning_delta` and one
    `complete` per response; a `complete` reports the `status` the response ended in, what it cost in
    tokens and how long the caller waited for the first of them. An `interrupt` frame naming
    `response_ids` abandons responses still being generated, which still settle and are still billed for
    what they produced before being cut off. All three report failures as `error` frames and end with
    `closed`.
    Search is answered at `/v1/search` rather than here: one question and its answer need no socket held
    open between them. Memory and phone are recorded rather than routed, so they are not served either.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error]
    """

    kwargs = _get_kwargs(
        modality=modality,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
) -> Any | Error | None:
    """Route one modality over a socket, for a pipeline running elsewhere

     A WebSocket, which OpenAPI cannot describe past the upgrade. This is the routing the agent does,
    offered a piece at a time: a caller running its own pipeline sends audio or text and gets
    transcripts, audio or completions back, and the request is failed over and billed exactly as it
    would be inside a session.
    Every socket opens with a `start` frame. It names either a `config_id`, a stored router config to
    take the options from, or the options outright; naming both overrides that config field by field.
    What it may carry is the modality's own option block - `SttOptions` for speech-to-text, `TtsOptions`
    for a voice, `LlmOptions` for a model - plus `agent_id` and `call_id` to attribute the work to a
    conversation and `tags` to bill it.
    Speech-to-text then takes binary PCM at the `sample_rate` the start frame named, 16 kHz mono by
    default, and returns `transcript` frames. Text-to-speech takes `speak` frames and returns binary
    audio with `synthesis_complete` between utterances: each audio frame opens with a little-endian
    header of a uint32 sample rate, a uint16 channel count and two reserved bytes, followed by PCM16
    samples. Language models take `respond` frames, each naming an `id` and anything from `LlmOptions`
    for that one response along with its `tools`, and return `delta`, `reasoning_delta` and one
    `complete` per response; a `complete` reports the `status` the response ended in, what it cost in
    tokens and how long the caller waited for the first of them. An `interrupt` frame naming
    `response_ids` abandons responses still being generated, which still settle and are still billed for
    what they produced before being cut off. All three report failures as `error` frames and end with
    `closed`.
    Search is answered at `/v1/search` rather than here: one question and its answer need no socket held
    open between them. Memory and phone are recorded rather than routed, so they are not served either.

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error
    """

    return (
        await asyncio_detailed(
            modality=modality,
            client=client,
        )
    ).parsed
