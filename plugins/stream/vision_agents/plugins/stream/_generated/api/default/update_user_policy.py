from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.user_policy import UserPolicy
from ...models.user_policy_input import UserPolicyInput
from ...types import Response


def _get_kwargs(
    user_id: str,
    *,
    body: UserPolicyInput,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/policies/users/{user_id}".format(
            user_id=quote(str(user_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | UserPolicy | None:
    if response.status_code == 200:
        response_200 = UserPolicy.from_dict(response.json())

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
) -> Response[ErrorResponse | UserPolicy]:
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
    body: UserPolicyInput,
) -> Response[ErrorResponse | UserPolicy]:
    """Replace one end user's own quota

     The user's own quota wins over the app's, the organization's and the router's default, each number
    on its own, and counts the app's backend acting for the user as well as the user's devices. A quota
    left out, null, or setting neither number, removes the user's own. A user's policy holds a quota
    only, so any other field is a 400. The router that wrote it holds to it at once, every other within
    5 s.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The end user, by the id the app's tokens name them with.
        body (UserPolicyInput): One end user's own policy, which holds a quota only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | UserPolicy]
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
    body: UserPolicyInput,
) -> ErrorResponse | UserPolicy | None:
    """Replace one end user's own quota

     The user's own quota wins over the app's, the organization's and the router's default, each number
    on its own, and counts the app's backend acting for the user as well as the user's devices. A quota
    left out, null, or setting neither number, removes the user's own. A user's policy holds a quota
    only, so any other field is a 400. The router that wrote it holds to it at once, every other within
    5 s.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The end user, by the id the app's tokens name them with.
        body (UserPolicyInput): One end user's own policy, which holds a quota only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | UserPolicy
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
    body: UserPolicyInput,
) -> Response[ErrorResponse | UserPolicy]:
    """Replace one end user's own quota

     The user's own quota wins over the app's, the organization's and the router's default, each number
    on its own, and counts the app's backend acting for the user as well as the user's devices. A quota
    left out, null, or setting neither number, removes the user's own. A user's policy holds a quota
    only, so any other field is a 400. The router that wrote it holds to it at once, every other within
    5 s.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The end user, by the id the app's tokens name them with.
        body (UserPolicyInput): One end user's own policy, which holds a quota only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | UserPolicy]
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
    body: UserPolicyInput,
) -> ErrorResponse | UserPolicy | None:
    """Replace one end user's own quota

     The user's own quota wins over the app's, the organization's and the router's default, each number
    on its own, and counts the app's backend acting for the user as well as the user's devices. A quota
    left out, null, or setting neither number, removes the user's own. A user's policy holds a quota
    only, so any other field is a 400. The router that wrote it holds to it at once, every other within
    5 s.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        user_id (str): The end user, by the id the app's tokens name them with.
        body (UserPolicyInput): One end user's own policy, which holds a quota only.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | UserPolicy
    """

    return (
        await asyncio_detailed(
            user_id=user_id,
            client=client,
            body=body,
        )
    ).parsed
