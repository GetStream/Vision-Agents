from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.session import Session
from ...models.update_session_request import UpdateSessionRequest
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: UpdateSessionRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/v1/agents/sessions/{id}".format(
            id=quote(str(id), safe=""),
        ),
    }

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
    body: UpdateSessionRequest,
) -> Response[ErrorResponse | Session]:
    """Change a session

     Renames a session, relabels it or moves it onto other models, for this session only: the agent
    config it started from is untouched. A field left out is left as it is. The id, the call, incognito
    and the instructions are what the session is, so they cannot change: the instructions are the agent
    config's, and forking is how to get a session that differs in the rest.

    An end user's device may change a session's title, description and custom, so a person can tidy up
    their own conversations. Models and voice are the backend's to change, and a device asking for them
    is refused with a 403.

    A session that ended can still be renamed and relabelled. Models only mean something to a session
    that is running, so asking to change them on one that ended is refused.

    Model changes are opened before anything changes, so a target that does not route is refused and the
    session carries on as it was. Models take over from the next turn; a reply being spoken finishes on
    what it started with. Naming sts makes the session native, and an empty sts makes it a cascade
    again. A session that started with the person's episode cards cannot be moved onto a speech-to-
    speech model: 400, carded_session_to_native. A title or description given here stops the router
    naming the conversation for what was said.

    Args:
        id (str): The session, as returned when it was created.
        body (UpdateSessionRequest): What to change about one session. A field left out is left as
            it is. Title, description and custom can change on a session that ended, and are all an
            end user's device may change; everything else needs the session running and a server-side
            caller.

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
    body: UpdateSessionRequest,
) -> ErrorResponse | Session | None:
    """Change a session

     Renames a session, relabels it or moves it onto other models, for this session only: the agent
    config it started from is untouched. A field left out is left as it is. The id, the call, incognito
    and the instructions are what the session is, so they cannot change: the instructions are the agent
    config's, and forking is how to get a session that differs in the rest.

    An end user's device may change a session's title, description and custom, so a person can tidy up
    their own conversations. Models and voice are the backend's to change, and a device asking for them
    is refused with a 403.

    A session that ended can still be renamed and relabelled. Models only mean something to a session
    that is running, so asking to change them on one that ended is refused.

    Model changes are opened before anything changes, so a target that does not route is refused and the
    session carries on as it was. Models take over from the next turn; a reply being spoken finishes on
    what it started with. Naming sts makes the session native, and an empty sts makes it a cascade
    again. A session that started with the person's episode cards cannot be moved onto a speech-to-
    speech model: 400, carded_session_to_native. A title or description given here stops the router
    naming the conversation for what was said.

    Args:
        id (str): The session, as returned when it was created.
        body (UpdateSessionRequest): What to change about one session. A field left out is left as
            it is. Title, description and custom can change on a session that ended, and are all an
            end user's device may change; everything else needs the session running and a server-side
            caller.

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
    body: UpdateSessionRequest,
) -> Response[ErrorResponse | Session]:
    """Change a session

     Renames a session, relabels it or moves it onto other models, for this session only: the agent
    config it started from is untouched. A field left out is left as it is. The id, the call, incognito
    and the instructions are what the session is, so they cannot change: the instructions are the agent
    config's, and forking is how to get a session that differs in the rest.

    An end user's device may change a session's title, description and custom, so a person can tidy up
    their own conversations. Models and voice are the backend's to change, and a device asking for them
    is refused with a 403.

    A session that ended can still be renamed and relabelled. Models only mean something to a session
    that is running, so asking to change them on one that ended is refused.

    Model changes are opened before anything changes, so a target that does not route is refused and the
    session carries on as it was. Models take over from the next turn; a reply being spoken finishes on
    what it started with. Naming sts makes the session native, and an empty sts makes it a cascade
    again. A session that started with the person's episode cards cannot be moved onto a speech-to-
    speech model: 400, carded_session_to_native. A title or description given here stops the router
    naming the conversation for what was said.

    Args:
        id (str): The session, as returned when it was created.
        body (UpdateSessionRequest): What to change about one session. A field left out is left as
            it is. Title, description and custom can change on a session that ended, and are all an
            end user's device may change; everything else needs the session running and a server-side
            caller.

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
    body: UpdateSessionRequest,
) -> ErrorResponse | Session | None:
    """Change a session

     Renames a session, relabels it or moves it onto other models, for this session only: the agent
    config it started from is untouched. A field left out is left as it is. The id, the call, incognito
    and the instructions are what the session is, so they cannot change: the instructions are the agent
    config's, and forking is how to get a session that differs in the rest.

    An end user's device may change a session's title, description and custom, so a person can tidy up
    their own conversations. Models and voice are the backend's to change, and a device asking for them
    is refused with a 403.

    A session that ended can still be renamed and relabelled. Models only mean something to a session
    that is running, so asking to change them on one that ended is refused.

    Model changes are opened before anything changes, so a target that does not route is refused and the
    session carries on as it was. Models take over from the next turn; a reply being spoken finishes on
    what it started with. Naming sts makes the session native, and an empty sts makes it a cascade
    again. A session that started with the person's episode cards cannot be moved onto a speech-to-
    speech model: 400, carded_session_to_native. A title or description given here stops the router
    naming the conversation for what was said.

    Args:
        id (str): The session, as returned when it was created.
        body (UpdateSessionRequest): What to change about one session. A field left out is left as
            it is. Title, description and custom can change on a session that ended, and are all an
            end user's device may change; everything else needs the session running and a server-side
            caller.

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
