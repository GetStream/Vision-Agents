from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.call_token import CallToken
from ...models.call_token_request import CallTokenRequest
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    body: CallTokenRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/calls/{id}/token".format(
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
) -> CallToken | Error | None:
    if response.status_code == 200:
        response_200 = CallToken.from_dict(response.json())

        return response_200

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
) -> Response[CallToken | Error]:
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
    body: CallTokenRequest | Unset = UNSET,
) -> Response[CallToken | Error]:
    """What a browser needs to join this call

     Mints a Stream token so a person can join the call from a browser and talk to the agent, and says
    which call to join with it. The secret stays here: the browser is handed a token that expires, never
    the key that signs one.

    Args:
        id (str):
        body (CallTokenRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CallToken | Error]
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
    body: CallTokenRequest | Unset = UNSET,
) -> CallToken | Error | None:
    """What a browser needs to join this call

     Mints a Stream token so a person can join the call from a browser and talk to the agent, and says
    which call to join with it. The secret stays here: the browser is handed a token that expires, never
    the key that signs one.

    Args:
        id (str):
        body (CallTokenRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CallToken | Error
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
    body: CallTokenRequest | Unset = UNSET,
) -> Response[CallToken | Error]:
    """What a browser needs to join this call

     Mints a Stream token so a person can join the call from a browser and talk to the agent, and says
    which call to join with it. The secret stays here: the browser is handed a token that expires, never
    the key that signs one.

    Args:
        id (str):
        body (CallTokenRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CallToken | Error]
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
    body: CallTokenRequest | Unset = UNSET,
) -> CallToken | Error | None:
    """What a browser needs to join this call

     Mints a Stream token so a person can join the call from a browser and talk to the agent, and says
    which call to join with it. The secret stays here: the browser is handed a token that expires, never
    the key that signs one.

    Args:
        id (str):
        body (CallTokenRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CallToken | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
