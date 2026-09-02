from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.prepare_voice_request import PrepareVoiceRequest
from ...models.voice import Voice
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: PrepareVoiceRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/voices/{id}/prepare".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | Voice | None:
    if response.status_code == 200:
        response_200 = Voice.from_dict(response.json())

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
) -> Response[Error | Voice]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: PrepareVoiceRequest,
) -> Response[Error | Voice]:
    """Teach the text-to-speech providers this voice

     Each provider is sent the recordings and hands back an id of its own, which is remembered so a
    session can ask for this voice by name. Providers are prepared independently: one refusing the
    recordings leaves the others usable, and the binding says why it refused.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (PrepareVoiceRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Voice]
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
    client: AuthenticatedClient | Client,
    body: PrepareVoiceRequest,
) -> Error | Voice | None:
    """Teach the text-to-speech providers this voice

     Each provider is sent the recordings and hands back an id of its own, which is remembered so a
    session can ask for this voice by name. Providers are prepared independently: one refusing the
    recordings leaves the others usable, and the binding says why it refused.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (PrepareVoiceRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Voice
    """

    return sync_detailed(
        id=id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: PrepareVoiceRequest,
) -> Response[Error | Voice]:
    """Teach the text-to-speech providers this voice

     Each provider is sent the recordings and hands back an id of its own, which is remembered so a
    session can ask for this voice by name. Providers are prepared independently: one refusing the
    recordings leaves the others usable, and the binding says why it refused.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (PrepareVoiceRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Voice]
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
    client: AuthenticatedClient | Client,
    body: PrepareVoiceRequest,
) -> Error | Voice | None:
    """Teach the text-to-speech providers this voice

     Each provider is sent the recordings and hands back an id of its own, which is remembered so a
    session can ask for this voice by name. Providers are prepared independently: one refusing the
    recordings leaves the others usable, and the binding says why it refused.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        body (PrepareVoiceRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Voice
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
