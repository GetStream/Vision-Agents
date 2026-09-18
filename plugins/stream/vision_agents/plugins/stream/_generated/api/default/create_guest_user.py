from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.guest_user import GuestUser
from ...models.guest_user_request import GuestUserRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: GuestUserRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/guests",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | Error | GuestUser | None:
    if response.status_code == 201:
        response_201 = GuestUser.from_dict(response.json())

        return response_201

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = cast(Any, None)
        return response_403

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | Error | GuestUser]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: GuestUserRequest | Unset = UNSET,
) -> Response[Any | Error | GuestUser]:
    """Mint a guest so somebody can talk to an agent before signing up

     Returns a user id and a Stream token with role guest, which is what the chat and video SDKs connect
    with. From the router's point of view a guest is an ordinary end user whose name is worth less:
    their sessions are their own and nobody else's, but a guest id is not evidence of who anybody is, so
    a guest cannot be handed another guest's conversations by naming their id.
    Open to a page on purpose. A guest that a backend had to mint is a guest every anonymous visitor
    costs a round trip through the customer's own servers, which is exactly the integration this is
    meant to remove. An app that has turned guests off refuses it.

    Args:
        body (GuestUserRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error | GuestUser]
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
    body: GuestUserRequest | Unset = UNSET,
) -> Any | Error | GuestUser | None:
    """Mint a guest so somebody can talk to an agent before signing up

     Returns a user id and a Stream token with role guest, which is what the chat and video SDKs connect
    with. From the router's point of view a guest is an ordinary end user whose name is worth less:
    their sessions are their own and nobody else's, but a guest id is not evidence of who anybody is, so
    a guest cannot be handed another guest's conversations by naming their id.
    Open to a page on purpose. A guest that a backend had to mint is a guest every anonymous visitor
    costs a round trip through the customer's own servers, which is exactly the integration this is
    meant to remove. An app that has turned guests off refuses it.

    Args:
        body (GuestUserRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error | GuestUser
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: GuestUserRequest | Unset = UNSET,
) -> Response[Any | Error | GuestUser]:
    """Mint a guest so somebody can talk to an agent before signing up

     Returns a user id and a Stream token with role guest, which is what the chat and video SDKs connect
    with. From the router's point of view a guest is an ordinary end user whose name is worth less:
    their sessions are their own and nobody else's, but a guest id is not evidence of who anybody is, so
    a guest cannot be handed another guest's conversations by naming their id.
    Open to a page on purpose. A guest that a backend had to mint is a guest every anonymous visitor
    costs a round trip through the customer's own servers, which is exactly the integration this is
    meant to remove. An app that has turned guests off refuses it.

    Args:
        body (GuestUserRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | Error | GuestUser]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: GuestUserRequest | Unset = UNSET,
) -> Any | Error | GuestUser | None:
    """Mint a guest so somebody can talk to an agent before signing up

     Returns a user id and a Stream token with role guest, which is what the chat and video SDKs connect
    with. From the router's point of view a guest is an ordinary end user whose name is worth less:
    their sessions are their own and nobody else's, but a guest id is not evidence of who anybody is, so
    a guest cannot be handed another guest's conversations by naming their id.
    Open to a page on purpose. A guest that a backend had to mint is a guest every anonymous visitor
    costs a round trip through the customer's own servers, which is exactly the integration this is
    meant to remove. An app that has turned guests off refuses it.

    Args:
        body (GuestUserRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | Error | GuestUser
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
