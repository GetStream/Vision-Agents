from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_user import AgentUser
from ...models.agent_user_request import AgentUserRequest
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    user_id: str,
    *,
    body: AgentUserRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/users/{user_id}".format(
            user_id=quote(str(user_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentUser | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = AgentUser.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[AgentUser | ErrorResponse]:
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
    body: AgentUserRequest,
) -> Response[AgentUser | ErrorResponse]:
    """Link a user to a phone number

     Says that a user of the app and a phone number are one person, so the person's app chat and what
    they text and call from that number from then on share one omni-channel with each agent of the app.
    What the number said before the link stays apart: under an agent with episode_cards record or on
    when the link is made, the number's earlier conversations, and one still going on, keep the omni-
    channel they had, and the user's app chat reads none of their cards. Otherwise, where the person
    already had one of each, both point at the older one, and the user still reads none of the number's
    cards from before the link once the agent turns episode_cards on. include_phone_history and
    phone_history_since say how much of the number's past the user reads instead: all of it, or what it
    said from a time on, in its cards and in the sessions on its threads. No message moves. One number
    for each user, and one user for each number, within the app: the same user id in another Stream app
    is another person. Sending the same number again with neither field changes nothing; null unlinks.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The Stream user id the app chat knows the user by: the id the app's own
            token signs.
        body (AgentUserRequest): Who one of the app's users is, as the app's backend knows it.
            phone is required: a number, or null for none.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentUser | ErrorResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    user_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AgentUserRequest,
) -> AgentUser | ErrorResponse | None:
    """Link a user to a phone number

     Says that a user of the app and a phone number are one person, so the person's app chat and what
    they text and call from that number from then on share one omni-channel with each agent of the app.
    What the number said before the link stays apart: under an agent with episode_cards record or on
    when the link is made, the number's earlier conversations, and one still going on, keep the omni-
    channel they had, and the user's app chat reads none of their cards. Otherwise, where the person
    already had one of each, both point at the older one, and the user still reads none of the number's
    cards from before the link once the agent turns episode_cards on. include_phone_history and
    phone_history_since say how much of the number's past the user reads instead: all of it, or what it
    said from a time on, in its cards and in the sessions on its threads. No message moves. One number
    for each user, and one user for each number, within the app: the same user id in another Stream app
    is another person. Sending the same number again with neither field changes nothing; null unlinks.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The Stream user id the app chat knows the user by: the id the app's own
            token signs.
        body (AgentUserRequest): Who one of the app's users is, as the app's backend knows it.
            phone is required: a number, or null for none.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentUser | ErrorResponse
    """

    return sync_detailed(
        user_id=user_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    user_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AgentUserRequest,
) -> Response[AgentUser | ErrorResponse]:
    """Link a user to a phone number

     Says that a user of the app and a phone number are one person, so the person's app chat and what
    they text and call from that number from then on share one omni-channel with each agent of the app.
    What the number said before the link stays apart: under an agent with episode_cards record or on
    when the link is made, the number's earlier conversations, and one still going on, keep the omni-
    channel they had, and the user's app chat reads none of their cards. Otherwise, where the person
    already had one of each, both point at the older one, and the user still reads none of the number's
    cards from before the link once the agent turns episode_cards on. include_phone_history and
    phone_history_since say how much of the number's past the user reads instead: all of it, or what it
    said from a time on, in its cards and in the sessions on its threads. No message moves. One number
    for each user, and one user for each number, within the app: the same user id in another Stream app
    is another person. Sending the same number again with neither field changes nothing; null unlinks.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The Stream user id the app chat knows the user by: the id the app's own
            token signs.
        body (AgentUserRequest): Who one of the app's users is, as the app's backend knows it.
            phone is required: a number, or null for none.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentUser | ErrorResponse]
    """

    kwargs = _get_kwargs(
        user_id=user_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    user_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AgentUserRequest,
) -> AgentUser | ErrorResponse | None:
    """Link a user to a phone number

     Says that a user of the app and a phone number are one person, so the person's app chat and what
    they text and call from that number from then on share one omni-channel with each agent of the app.
    What the number said before the link stays apart: under an agent with episode_cards record or on
    when the link is made, the number's earlier conversations, and one still going on, keep the omni-
    channel they had, and the user's app chat reads none of their cards. Otherwise, where the person
    already had one of each, both point at the older one, and the user still reads none of the number's
    cards from before the link once the agent turns episode_cards on. include_phone_history and
    phone_history_since say how much of the number's past the user reads instead: all of it, or what it
    said from a time on, in its cards and in the sessions on its threads. No message moves. One number
    for each user, and one user for each number, within the app: the same user id in another Stream app
    is another person. Sending the same number again with neither field changes nothing; null unlinks.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The Stream user id the app chat knows the user by: the id the app's own
            token signs.
        body (AgentUserRequest): Who one of the app's users is, as the app's backend knows it.
            phone is required: a number, or null for none.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentUser | ErrorResponse
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            body=body,
        )
    ).parsed
