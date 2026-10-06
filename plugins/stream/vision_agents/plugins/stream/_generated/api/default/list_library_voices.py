from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.library_voices import LibraryVoices
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    provider: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["provider"] = provider

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/voices/library",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | LibraryVoices | None:
    if response.status_code == 200:
        response_200 = LibraryVoices.from_dict(response.json())

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

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | LibraryVoices]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    provider: str | Unset = UNSET,
) -> Response[ErrorResponse | LibraryVoices]:
    """The voices the speech providers offer

     The catalogue each provider publishes, so a voice can be picked by name rather than by pasting an
    id. Only providers this deployment holds a key for and that publish a library appear; for the others
    a voice is still whatever the vendor's own terms call one, and has to be typed. A provider that
    cannot be reached is reported in `unavailable` rather than emptying the list.

    Args:
        provider (str | Unset): Only this provider's voices.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LibraryVoices]
    """

    kwargs = _get_kwargs(
        provider=provider,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    provider: str | Unset = UNSET,
) -> ErrorResponse | LibraryVoices | None:
    """The voices the speech providers offer

     The catalogue each provider publishes, so a voice can be picked by name rather than by pasting an
    id. Only providers this deployment holds a key for and that publish a library appear; for the others
    a voice is still whatever the vendor's own terms call one, and has to be typed. A provider that
    cannot be reached is reported in `unavailable` rather than emptying the list.

    Args:
        provider (str | Unset): Only this provider's voices.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LibraryVoices
    """

    return sync_detailed(
        client=client,
        provider=provider,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    provider: str | Unset = UNSET,
) -> Response[ErrorResponse | LibraryVoices]:
    """The voices the speech providers offer

     The catalogue each provider publishes, so a voice can be picked by name rather than by pasting an
    id. Only providers this deployment holds a key for and that publish a library appear; for the others
    a voice is still whatever the vendor's own terms call one, and has to be typed. A provider that
    cannot be reached is reported in `unavailable` rather than emptying the list.

    Args:
        provider (str | Unset): Only this provider's voices.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LibraryVoices]
    """

    kwargs = _get_kwargs(
        provider=provider,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    provider: str | Unset = UNSET,
) -> ErrorResponse | LibraryVoices | None:
    """The voices the speech providers offer

     The catalogue each provider publishes, so a voice can be picked by name rather than by pasting an
    id. Only providers this deployment holds a key for and that publish a library appear; for the others
    a voice is still whatever the vendor's own terms call one, and has to be typed. A provider that
    cannot be reached is reported in `unavailable` rather than emptying the list.

    Args:
        provider (str | Unset): Only this provider's voices.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LibraryVoices
    """

    return (
        await asyncio_detailed(
            client=client,
            provider=provider,
        )
    ).parsed
