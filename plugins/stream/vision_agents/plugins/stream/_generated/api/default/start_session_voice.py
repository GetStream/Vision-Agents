from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.session import Session
from ...types import Response


def _get_kwargs(
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/{id}/voice".format(
            id=quote(str(id), safe=""),
        ),
    }

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
) -> Response[ErrorResponse | Session]:
    """Start voice on a session

     The agent joins the call agent:<session id>, which joining creates, with the conversation so far,
    and returns once it is there. What is said on the call and what is typed into the session are one
    conversation, kept in the same Stream Chat channel: a typed question is answered aloud. The models
    are the ones the session was opened with, or the defaults, and a native config speaks with its
    speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing. A conversation in writing
    that ended is carried on under the same id, as a message to it is.

    Args:
        id (str): The session, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Session]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | Session | None:
    """Start voice on a session

     The agent joins the call agent:<session id>, which joining creates, with the conversation so far,
    and returns once it is there. What is said on the call and what is typed into the session are one
    conversation, kept in the same Stream Chat channel: a typed question is answered aloud. The models
    are the ones the session was opened with, or the defaults, and a native config speaks with its
    speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing. A conversation in writing
    that ended is carried on under the same id, as a message to it is.

    Args:
        id (str): The session, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Session
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ErrorResponse | Session]:
    """Start voice on a session

     The agent joins the call agent:<session id>, which joining creates, with the conversation so far,
    and returns once it is there. What is said on the call and what is typed into the session are one
    conversation, kept in the same Stream Chat channel: a typed question is answered aloud. The models
    are the ones the session was opened with, or the defaults, and a native config speaks with its
    speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing. A conversation in writing
    that ended is carried on under the same id, as a message to it is.

    Args:
        id (str): The session, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Session]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | Session | None:
    """Start voice on a session

     The agent joins the call agent:<session id>, which joining creates, with the conversation so far,
    and returns once it is there. What is said on the call and what is typed into the session are one
    conversation, kept in the same Stream Chat channel: a typed question is answered aloud. The models
    are the ones the session was opened with, or the defaults, and a native config speaks with its
    speech-to-speech model.

    Starting voice on a session that is already on its call changes nothing. A conversation in writing
    that ended is carried on under the same id, as a message to it is.

    Args:
        id (str): The session, as returned when it was created.

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
        )
    ).parsed
