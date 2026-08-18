from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_session_request import CreateSessionRequest
from ...models.error import Error
from ...models.session import Session
from ...types import Response


def _get_kwargs(
    *,
    body: CreateSessionRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | Session | None:
    if response.status_code == 201:
        response_201 = Session.from_dict(response.json())

        return response_201

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

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
) -> Response[Error | Session]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CreateSessionRequest,
) -> Response[Error | Session]:
    """Join a call as a voice agent

     The whole conversation runs here: the agent joins the call, transcribes what it hears, answers it
    and speaks back, all through the routers. The caller keeps the session id and watches the
    conversation over the events socket.
    It returns once the agent is in the call, so a session that comes back is one that is already
    listening. Tools declared here are the caller's own: the model asks for them over the events socket
    and waits for the caller to answer.

    Args:
        body (CreateSessionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Session]
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
    body: CreateSessionRequest,
) -> Error | Session | None:
    """Join a call as a voice agent

     The whole conversation runs here: the agent joins the call, transcribes what it hears, answers it
    and speaks back, all through the routers. The caller keeps the session id and watches the
    conversation over the events socket.
    It returns once the agent is in the call, so a session that comes back is one that is already
    listening. Tools declared here are the caller's own: the model asks for them over the events socket
    and waits for the caller to answer.

    Args:
        body (CreateSessionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Session
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CreateSessionRequest,
) -> Response[Error | Session]:
    """Join a call as a voice agent

     The whole conversation runs here: the agent joins the call, transcribes what it hears, answers it
    and speaks back, all through the routers. The caller keeps the session id and watches the
    conversation over the events socket.
    It returns once the agent is in the call, so a session that comes back is one that is already
    listening. Tools declared here are the caller's own: the model asks for them over the events socket
    and waits for the caller to answer.

    Args:
        body (CreateSessionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Session]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: CreateSessionRequest,
) -> Error | Session | None:
    """Join a call as a voice agent

     The whole conversation runs here: the agent joins the call, transcribes what it hears, answers it
    and speaks back, all through the routers. The caller keeps the session id and watches the
    conversation over the events socket.
    It returns once the agent is in the call, so a session that comes back is one that is already
    listening. Tools declared here are the caller's own: the model asks for them over the events socket
    and waits for the caller to answer.

    Args:
        body (CreateSessionRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Session
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
