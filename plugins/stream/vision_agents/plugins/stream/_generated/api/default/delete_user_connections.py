from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    user_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/agents/users/{user_id}/connections".format(
            user_id=quote(str(user_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    user_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Delete every connection of one user

     For offboarding and erasure requests: deletes every connection the user owns, live or deleted
    before, for good, with its credentials, its pending consents and its tool call log, so the user's id
    and their provider accounts' ids are gone. The next session for the user attaches none of them. The
    provider is not asked to revoke what it issued. The audit keeps one grant_revoked row for each
    connection that still held a grant, naming neither the user nor the account, and the audit rows of
    those connections lose their request, session and attempt ids. A user with no connections is not an
    error.

    It deletes connections, not sessions: the tool calls and audit rows of the app's own connections
    keep the ids of the user's sessions that caused them. Delete those sessions too, and each id names a
    session that no longer exists.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The user whose connections to delete, as owner.user_id named them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Delete every connection of one user

     For offboarding and erasure requests: deletes every connection the user owns, live or deleted
    before, for good, with its credentials, its pending consents and its tool call log, so the user's id
    and their provider accounts' ids are gone. The next session for the user attaches none of them. The
    provider is not asked to revoke what it issued. The audit keeps one grant_revoked row for each
    connection that still held a grant, naming neither the user nor the account, and the audit rows of
    those connections lose their request, session and attempt ids. A user with no connections is not an
    error.

    It deletes connections, not sessions: the tool calls and audit rows of the app's own connections
    keep the ids of the user's sessions that caused them. Delete those sessions too, and each id names a
    session that no longer exists.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The user whose connections to delete, as owner.user_id named them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Delete every connection of one user

     For offboarding and erasure requests: deletes every connection the user owns, live or deleted
    before, for good, with its credentials, its pending consents and its tool call log, so the user's id
    and their provider accounts' ids are gone. The next session for the user attaches none of them. The
    provider is not asked to revoke what it issued. The audit keeps one grant_revoked row for each
    connection that still held a grant, naming neither the user nor the account, and the audit rows of
    those connections lose their request, session and attempt ids. A user with no connections is not an
    error.

    It deletes connections, not sessions: the tool calls and audit rows of the app's own connections
    keep the ids of the user's sessions that caused them. Delete those sessions too, and each id names a
    session that no longer exists.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The user whose connections to delete, as owner.user_id named them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Delete every connection of one user

     For offboarding and erasure requests: deletes every connection the user owns, live or deleted
    before, for good, with its credentials, its pending consents and its tool call log, so the user's id
    and their provider accounts' ids are gone. The next session for the user attaches none of them. The
    provider is not asked to revoke what it issued. The audit keeps one grant_revoked row for each
    connection that still held a grant, naming neither the user nor the account, and the audit rows of
    those connections lose their request, session and attempt ids. A user with no connections is not an
    error.

    It deletes connections, not sessions: the tool calls and audit rows of the app's own connections
    keep the ids of the user's sessions that caused them. Delete those sessions too, and each id names a
    session that no longer exists.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The user whose connections to delete, as owner.user_id named them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
        )
    ).parsed
