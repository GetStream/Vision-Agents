from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.activated_branch import ActivatedBranch
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    response_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/{id}/responses/{response_id}/activate".format(
            id=quote(str(id), safe=""),
            response_id=quote(str(response_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ActivatedBranch | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ActivatedBranch.from_dict(response.json())

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

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 410:
        response_410 = ErrorResponse.from_dict(response.json())

        return response_410

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
) -> Response[ActivatedBranch | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ActivatedBranch | ErrorResponse]:
    """Make a hidden branch the conversation again

     The conversation from where the branch leaves it is hidden in Stream Chat and kept as a branch of
    its own, and the branch ending at the newest response under the one named is sent again: new
    messages, with new ids, sent with skip_push and carrying custom.resent_from, the id of the message
    each copies. The model answers from that branch from then on. Call it before asking anything while a
    hidden branch is shown.
    A branch already on the conversation changes nothing and answers already_active, so a retry is safe.
    rewound_to says where the channel was rewound to, which the acting device prunes its own view from:
    the message itself is gone too when inclusive.
    At most 100 messages are sent again; a longer branch is a 409 branch_too_long, and a conversation
    with no room left for another hidden range a 409 branch_limit_reached.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): A response of the branch to bring back. The branch ends at the newest
            response under it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ActivatedBranch | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        response_id=response_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ActivatedBranch | ErrorResponse | None:
    """Make a hidden branch the conversation again

     The conversation from where the branch leaves it is hidden in Stream Chat and kept as a branch of
    its own, and the branch ending at the newest response under the one named is sent again: new
    messages, with new ids, sent with skip_push and carrying custom.resent_from, the id of the message
    each copies. The model answers from that branch from then on. Call it before asking anything while a
    hidden branch is shown.
    A branch already on the conversation changes nothing and answers already_active, so a retry is safe.
    rewound_to says where the channel was rewound to, which the acting device prunes its own view from:
    the message itself is gone too when inclusive.
    At most 100 messages are sent again; a longer branch is a 409 branch_too_long, and a conversation
    with no room left for another hidden range a 409 branch_limit_reached.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): A response of the branch to bring back. The branch ends at the newest
            response under it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ActivatedBranch | ErrorResponse
    """

    return sync_detailed(
        id=id,
        response_id=response_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ActivatedBranch | ErrorResponse]:
    """Make a hidden branch the conversation again

     The conversation from where the branch leaves it is hidden in Stream Chat and kept as a branch of
    its own, and the branch ending at the newest response under the one named is sent again: new
    messages, with new ids, sent with skip_push and carrying custom.resent_from, the id of the message
    each copies. The model answers from that branch from then on. Call it before asking anything while a
    hidden branch is shown.
    A branch already on the conversation changes nothing and answers already_active, so a retry is safe.
    rewound_to says where the channel was rewound to, which the acting device prunes its own view from:
    the message itself is gone too when inclusive.
    At most 100 messages are sent again; a longer branch is a 409 branch_too_long, and a conversation
    with no room left for another hidden range a 409 branch_limit_reached.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): A response of the branch to bring back. The branch ends at the newest
            response under it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ActivatedBranch | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        response_id=response_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ActivatedBranch | ErrorResponse | None:
    """Make a hidden branch the conversation again

     The conversation from where the branch leaves it is hidden in Stream Chat and kept as a branch of
    its own, and the branch ending at the newest response under the one named is sent again: new
    messages, with new ids, sent with skip_push and carrying custom.resent_from, the id of the message
    each copies. The model answers from that branch from then on. Call it before asking anything while a
    hidden branch is shown.
    A branch already on the conversation changes nothing and answers already_active, so a retry is safe.
    rewound_to says where the channel was rewound to, which the acting device prunes its own view from:
    the message itself is gone too when inclusive.
    At most 100 messages are sent again; a longer branch is a 409 branch_too_long, and a conversation
    with no room left for another hidden range a 409 branch_limit_reached.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): A response of the branch to bring back. The branch ends at the newest
            response under it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ActivatedBranch | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            response_id=response_id,
            client=client,
        )
    ).parsed
