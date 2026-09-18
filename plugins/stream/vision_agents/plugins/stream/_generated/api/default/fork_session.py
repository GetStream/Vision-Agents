from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.fork_session_request import ForkSessionRequest
from ...models.session import Session
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    body: ForkSessionRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/{id}/fork".format(
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
    body: ForkSessionRequest | Unset = UNSET,
) -> Response[Error | Session]:
    """Continue a conversation as a new one

     Opens a session from another's spec, carrying its history across by default, and records where it
    came from. The usual reason is to ask the same question of a different model without losing the
    original answer, which is why anything in the request is written over what the parent was opened
    with.
    The parent is untouched and keeps running if it was running. Forking an incognito session is refused
    rather than answered with an empty conversation: there is nothing recorded to fork from, and
    pretending otherwise would hand back a session that quietly lost everything the caller thought they
    were continuing.

    Args:
        id (str):
        body (ForkSessionRequest | Unset): Continue a conversation as a new one. Everything the
            parent was opened with is inherited; anything named here is written over it, which is what
            makes a fork useful rather than a copy -- the usual reason to fork is to ask the same
            question of a different model.

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
    body: ForkSessionRequest | Unset = UNSET,
) -> Error | Session | None:
    """Continue a conversation as a new one

     Opens a session from another's spec, carrying its history across by default, and records where it
    came from. The usual reason is to ask the same question of a different model without losing the
    original answer, which is why anything in the request is written over what the parent was opened
    with.
    The parent is untouched and keeps running if it was running. Forking an incognito session is refused
    rather than answered with an empty conversation: there is nothing recorded to fork from, and
    pretending otherwise would hand back a session that quietly lost everything the caller thought they
    were continuing.

    Args:
        id (str):
        body (ForkSessionRequest | Unset): Continue a conversation as a new one. Everything the
            parent was opened with is inherited; anything named here is written over it, which is what
            makes a fork useful rather than a copy -- the usual reason to fork is to ask the same
            question of a different model.

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
    body: ForkSessionRequest | Unset = UNSET,
) -> Response[Error | Session]:
    """Continue a conversation as a new one

     Opens a session from another's spec, carrying its history across by default, and records where it
    came from. The usual reason is to ask the same question of a different model without losing the
    original answer, which is why anything in the request is written over what the parent was opened
    with.
    The parent is untouched and keeps running if it was running. Forking an incognito session is refused
    rather than answered with an empty conversation: there is nothing recorded to fork from, and
    pretending otherwise would hand back a session that quietly lost everything the caller thought they
    were continuing.

    Args:
        id (str):
        body (ForkSessionRequest | Unset): Continue a conversation as a new one. Everything the
            parent was opened with is inherited; anything named here is written over it, which is what
            makes a fork useful rather than a copy -- the usual reason to fork is to ask the same
            question of a different model.

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
    body: ForkSessionRequest | Unset = UNSET,
) -> Error | Session | None:
    """Continue a conversation as a new one

     Opens a session from another's spec, carrying its history across by default, and records where it
    came from. The usual reason is to ask the same question of a different model without losing the
    original answer, which is why anything in the request is written over what the parent was opened
    with.
    The parent is untouched and keeps running if it was running. Forking an incognito session is refused
    rather than answered with an empty conversation: there is nothing recorded to fork from, and
    pretending otherwise would hand back a session that quietly lost everything the caller thought they
    were continuing.

    Args:
        id (str):
        body (ForkSessionRequest | Unset): Continue a conversation as a new one. Everything the
            parent was opened with is inherited; anything named here is written over it, which is what
            makes a fork useful rather than a copy -- the usual reason to fork is to ask the same
            question of a different model.

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
