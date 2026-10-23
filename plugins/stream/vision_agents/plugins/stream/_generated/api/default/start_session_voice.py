from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.session import Session
from ...models.start_session_voice_request import StartSessionVoiceRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    body: StartSessionVoiceRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/{id}/voice".format(
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
) -> ErrorResponse | Session | None:
    if response.status_code == 200:
        response_200 = Session.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | Session]:
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
    body: StartSessionVoiceRequest | Unset = UNSET,
) -> Response[ErrorResponse | Session]:
    """Start voice on a session

     The agent joins the call named in call, else the one voice was last on, else agent:<session id>,
    which joining creates, with the conversation so far, and returns once it is there. What is said on
    the call and what is typed into the session are one conversation, kept in the same Stream Chat
    channel: a typed question is answered aloud. The models are the ones the session was opened with, or
    the defaults, and a native config speaks with its speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing, and naming another call
    while it is on is refused. A conversation in writing that ended is carried on under the same id, as
    a message to it is. Voice on a socket session (/v1/agents/socket) that was stopped is refused: the
    socket was its call, and stopping voice hung it up, so open a new socket.

    Args:
        id (str): The session, as returned when it was created.
        body (StartSessionVoiceRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Session]
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
    body: StartSessionVoiceRequest | Unset = UNSET,
) -> ErrorResponse | Session | None:
    """Start voice on a session

     The agent joins the call named in call, else the one voice was last on, else agent:<session id>,
    which joining creates, with the conversation so far, and returns once it is there. What is said on
    the call and what is typed into the session are one conversation, kept in the same Stream Chat
    channel: a typed question is answered aloud. The models are the ones the session was opened with, or
    the defaults, and a native config speaks with its speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing, and naming another call
    while it is on is refused. A conversation in writing that ended is carried on under the same id, as
    a message to it is. Voice on a socket session (/v1/agents/socket) that was stopped is refused: the
    socket was its call, and stopping voice hung it up, so open a new socket.

    Args:
        id (str): The session, as returned when it was created.
        body (StartSessionVoiceRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Session
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
    body: StartSessionVoiceRequest | Unset = UNSET,
) -> Response[ErrorResponse | Session]:
    """Start voice on a session

     The agent joins the call named in call, else the one voice was last on, else agent:<session id>,
    which joining creates, with the conversation so far, and returns once it is there. What is said on
    the call and what is typed into the session are one conversation, kept in the same Stream Chat
    channel: a typed question is answered aloud. The models are the ones the session was opened with, or
    the defaults, and a native config speaks with its speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing, and naming another call
    while it is on is refused. A conversation in writing that ended is carried on under the same id, as
    a message to it is. Voice on a socket session (/v1/agents/socket) that was stopped is refused: the
    socket was its call, and stopping voice hung it up, so open a new socket.

    Args:
        id (str): The session, as returned when it was created.
        body (StartSessionVoiceRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Session]
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
    body: StartSessionVoiceRequest | Unset = UNSET,
) -> ErrorResponse | Session | None:
    """Start voice on a session

     The agent joins the call named in call, else the one voice was last on, else agent:<session id>,
    which joining creates, with the conversation so far, and returns once it is there. What is said on
    the call and what is typed into the session are one conversation, kept in the same Stream Chat
    channel: a typed question is answered aloud. The models are the ones the session was opened with, or
    the defaults, and a native config speaks with its speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing, and naming another call
    while it is on is refused. A conversation in writing that ended is carried on under the same id, as
    a message to it is. Voice on a socket session (/v1/agents/socket) that was stopped is refused: the
    socket was its call, and stopping voice hung it up, so open a new socket.

    Args:
        id (str): The session, as returned when it was created.
        body (StartSessionVoiceRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Session
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
