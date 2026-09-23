from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.session import Session
from ...models.session_settings_request import SessionSettingsRequest
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: SessionSettingsRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/v1/agents/sessions/{id}/settings".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | Session | None:
    if response.status_code == 200:
        response_200 = Session.from_dict(response.json())

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
) -> Response[Error | Session]:
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
    body: SessionSettingsRequest,
) -> Response[Error | Session]:
    """Change the models and voice of one running session

     Swaps what the agent runs on without leaving the call, for this session only: the agent config it
    started from is untouched. The new models are opened before anything changes, so a target that does
    not route is refused and the agent carries on as it was. They take over from the next turn; a reply
    being spoken finishes on the models it started with.
    Naming sts makes the session native, and an empty sts makes it a cascade again, on whatever llm, stt
    and tts it names or had before. The conversation carries across: a conversation model is handed the
    history on every turn, and a speech-to-speech model is opened with the recent transcript in its
    instructions.

    Args:
        id (str):
        body (SessionSettingsRequest): What to change about one running session's models. A field
            left out is left as it is. The same safe knobs as ModelOverwrites, plus the voice.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Session]
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
    body: SessionSettingsRequest,
) -> Error | Session | None:
    """Change the models and voice of one running session

     Swaps what the agent runs on without leaving the call, for this session only: the agent config it
    started from is untouched. The new models are opened before anything changes, so a target that does
    not route is refused and the agent carries on as it was. They take over from the next turn; a reply
    being spoken finishes on the models it started with.
    Naming sts makes the session native, and an empty sts makes it a cascade again, on whatever llm, stt
    and tts it names or had before. The conversation carries across: a conversation model is handed the
    history on every turn, and a speech-to-speech model is opened with the recent transcript in its
    instructions.

    Args:
        id (str):
        body (SessionSettingsRequest): What to change about one running session's models. A field
            left out is left as it is. The same safe knobs as ModelOverwrites, plus the voice.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Session
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
    body: SessionSettingsRequest,
) -> Response[Error | Session]:
    """Change the models and voice of one running session

     Swaps what the agent runs on without leaving the call, for this session only: the agent config it
    started from is untouched. The new models are opened before anything changes, so a target that does
    not route is refused and the agent carries on as it was. They take over from the next turn; a reply
    being spoken finishes on the models it started with.
    Naming sts makes the session native, and an empty sts makes it a cascade again, on whatever llm, stt
    and tts it names or had before. The conversation carries across: a conversation model is handed the
    history on every turn, and a speech-to-speech model is opened with the recent transcript in its
    instructions.

    Args:
        id (str):
        body (SessionSettingsRequest): What to change about one running session's models. A field
            left out is left as it is. The same safe knobs as ModelOverwrites, plus the voice.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Session]
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
    body: SessionSettingsRequest,
) -> Error | Session | None:
    """Change the models and voice of one running session

     Swaps what the agent runs on without leaving the call, for this session only: the agent config it
    started from is untouched. The new models are opened before anything changes, so a target that does
    not route is refused and the agent carries on as it was. They take over from the next turn; a reply
    being spoken finishes on the models it started with.
    Naming sts makes the session native, and an empty sts makes it a cascade again, on whatever llm, stt
    and tts it names or had before. The conversation carries across: a conversation model is handed the
    history on every turn, and a speech-to-speech model is opened with the recent transcript in its
    instructions.

    Args:
        id (str):
        body (SessionSettingsRequest): What to change about one running session's models. A field
            left out is left as it is. The same safe knobs as ModelOverwrites, plus the voice.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Session
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
