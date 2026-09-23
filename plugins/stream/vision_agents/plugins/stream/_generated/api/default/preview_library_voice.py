from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.voice_preview import VoicePreview
from ...types import Response


def _get_kwargs(
    provider: str,
    voice: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/voices/library/{provider}/{voice}/preview".format(
            provider=quote(str(provider), safe=""),
            voice=quote(str(voice), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | VoicePreview | None:
    if response.status_code == 200:
        response_200 = VoicePreview.from_dict(response.json())

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
) -> Response[Error | VoicePreview]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    provider: str,
    voice: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | VoicePreview]:
    """Hear a voice from a provider's library

     The sample the vendor already published, fetched through the router because two of them want the
    deployment's key to hand it over. Nothing is synthesised, so browsing a library spends no credits.

    Args:
        provider (str):
        voice (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | VoicePreview]
    """

    kwargs = _get_kwargs(
        provider=provider,
        voice=voice,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    provider: str,
    voice: str,
    *,
    client: AuthenticatedClient | Client,
) -> Error | VoicePreview | None:
    """Hear a voice from a provider's library

     The sample the vendor already published, fetched through the router because two of them want the
    deployment's key to hand it over. Nothing is synthesised, so browsing a library spends no credits.

    Args:
        provider (str):
        voice (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | VoicePreview
    """

    return sync_detailed(
        provider=provider,
        voice=voice,
        client=client,
    ).parsed


async def asyncio_detailed(
    provider: str,
    voice: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | VoicePreview]:
    """Hear a voice from a provider's library

     The sample the vendor already published, fetched through the router because two of them want the
    deployment's key to hand it over. Nothing is synthesised, so browsing a library spends no credits.

    Args:
        provider (str):
        voice (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | VoicePreview]
    """

    kwargs = _get_kwargs(
        provider=provider,
        voice=voice,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    provider: str,
    voice: str,
    *,
    client: AuthenticatedClient | Client,
) -> Error | VoicePreview | None:
    """Hear a voice from a provider's library

     The sample the vendor already published, fetched through the router because two of them want the
    deployment's key to hand it over. Nothing is synthesised, so browsing a library spends no credits.

    Args:
        provider (str):
        voice (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | VoicePreview
    """

    return (
        await asyncio_detailed(
            provider=provider,
            voice=voice,
            client=client,
        )
    ).parsed
