from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    token: str,
    *,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["hub.mode"] = hub_mode

    params["hub.verify_token"] = hub_verify_token

    params["hub.challenge"] = hub_challenge

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/channels/hooks/{token}".format(
            token=quote(str(token), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | str | None:
    if response.status_code == 200:
        response_200 = response.text
        return response_200

    if response.status_code == 403:
        response_403 = cast(Any, None)
        return response_403

    if response.status_code == 410:
        response_410 = cast(Any, None)
        return response_410

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | ErrorResponse | str]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    token: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Response[Any | ErrorResponse | str]:
    """Answer a channel provider's webhook check

     What WhatsApp asks for before it will deliver: the verify token the line was connected with,
    answered with the challenge it sent, as text. Unauthenticated because Meta is not a customer; the
    token in the path names the line.

    Args:
        token (str):
        hub_mode (str | Unset):
        hub_verify_token (str | Unset):
        hub_challenge (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        token=token,
        hub_mode=hub_mode,
        hub_verify_token=hub_verify_token,
        hub_challenge=hub_challenge,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    token: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Any | ErrorResponse | str | None:
    """Answer a channel provider's webhook check

     What WhatsApp asks for before it will deliver: the verify token the line was connected with,
    answered with the challenge it sent, as text. Unauthenticated because Meta is not a customer; the
    token in the path names the line.

    Args:
        token (str):
        hub_mode (str | Unset):
        hub_verify_token (str | Unset):
        hub_challenge (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse | str
    """

    return sync_detailed(
        token=token,
        client=client,
        hub_mode=hub_mode,
        hub_verify_token=hub_verify_token,
        hub_challenge=hub_challenge,
    ).parsed


async def asyncio_detailed(
    token: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Response[Any | ErrorResponse | str]:
    """Answer a channel provider's webhook check

     What WhatsApp asks for before it will deliver: the verify token the line was connected with,
    answered with the challenge it sent, as text. Unauthenticated because Meta is not a customer; the
    token in the path names the line.

    Args:
        token (str):
        hub_mode (str | Unset):
        hub_verify_token (str | Unset):
        hub_challenge (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        token=token,
        hub_mode=hub_mode,
        hub_verify_token=hub_verify_token,
        hub_challenge=hub_challenge,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    token: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Any | ErrorResponse | str | None:
    """Answer a channel provider's webhook check

     What WhatsApp asks for before it will deliver: the verify token the line was connected with,
    answered with the challenge it sent, as text. Unauthenticated because Meta is not a customer; the
    token in the path names the line.

    Args:
        token (str):
        hub_mode (str | Unset):
        hub_verify_token (str | Unset):
        hub_challenge (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse | str
    """

    return (
        await asyncio_detailed(
            token=token,
            client=client,
            hub_mode=hub_mode,
            hub_verify_token=hub_verify_token,
            hub_challenge=hub_challenge,
        )
    ).parsed
