from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.call_tokens import CallTokens
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/calls/{id}/tokens".format(
            id=quote(str(id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> CallTokens | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = CallTokens.from_dict(response.json())

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
) -> Response[CallTokens | ErrorResponse]:
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
) -> Response[CallTokens | ErrorResponse]:
    """What a call's models read and wrote, and what their prompts were made of

     Read while the call is going as well as after it, unlike the usage on the call, which is counted
    once it is over. A prompt's parts are estimated: no provider says how much of a prompt was
    instructions, tools or images, so the router estimates it from each request and scales it to what
    the provider counted. The parts sum to the input tokens.

    Args:
        id (str): The resource, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CallTokens | ErrorResponse]
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
) -> CallTokens | ErrorResponse | None:
    """What a call's models read and wrote, and what their prompts were made of

     Read while the call is going as well as after it, unlike the usage on the call, which is counted
    once it is over. A prompt's parts are estimated: no provider says how much of a prompt was
    instructions, tools or images, so the router estimates it from each request and scales it to what
    the provider counted. The parts sum to the input tokens.

    Args:
        id (str): The resource, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CallTokens | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[CallTokens | ErrorResponse]:
    """What a call's models read and wrote, and what their prompts were made of

     Read while the call is going as well as after it, unlike the usage on the call, which is counted
    once it is over. A prompt's parts are estimated: no provider says how much of a prompt was
    instructions, tools or images, so the router estimates it from each request and scales it to what
    the provider counted. The parts sum to the input tokens.

    Args:
        id (str): The resource, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CallTokens | ErrorResponse]
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
) -> CallTokens | ErrorResponse | None:
    """What a call's models read and wrote, and what their prompts were made of

     Read while the call is going as well as after it, unlike the usage on the call, which is counted
    once it is over. A prompt's parts are estimated: no provider says how much of a prompt was
    instructions, tools or images, so the router estimates it from each request and scales it to what
    the provider counted. The parts sum to the input tokens.

    Args:
        id (str): The resource, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CallTokens | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
