from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    state: str | Unset = UNSET,
    code: str | Unset = UNSET,
    iss: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["state"] = state

    params["code"] = code

    params["iss"] = iss

    params["error"] = error

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/connectors/oauth/callback",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 302:
        response_302 = cast(Any, None)
        return response_302

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 403:
        response_403 = cast(Any, None)
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
    *,
    client: AuthenticatedClient | Client,
    state: str | Unset = UNSET,
    code: str | Unset = UNSET,
    iss: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
    """Finish a consent

     The redirect URI a provider sends the browser back to. The state must name an open consent, the
    browser must hold the cookie the handoff set, and the consent is used once. The router then
    exchanges the code and sends the browser to the dashboard with connection_id and status: connected,
    denied, failed, or account_mismatch when a reconnect came back with another provider account and the
    old grant was kept. Unauthenticated because the browser arrives from the provider.

    Args:
        state (str | Unset):
        code (str | Unset):
        iss (str | Unset):
        error (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        state=state,
        code=code,
        iss=iss,
        error=error,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    state: str | Unset = UNSET,
    code: str | Unset = UNSET,
    iss: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Any | ErrorResponse | None:
    """Finish a consent

     The redirect URI a provider sends the browser back to. The state must name an open consent, the
    browser must hold the cookie the handoff set, and the consent is used once. The router then
    exchanges the code and sends the browser to the dashboard with connection_id and status: connected,
    denied, failed, or account_mismatch when a reconnect came back with another provider account and the
    old grant was kept. Unauthenticated because the browser arrives from the provider.

    Args:
        state (str | Unset):
        code (str | Unset):
        iss (str | Unset):
        error (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        client=client,
        state=state,
        code=code,
        iss=iss,
        error=error,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    state: str | Unset = UNSET,
    code: str | Unset = UNSET,
    iss: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
    """Finish a consent

     The redirect URI a provider sends the browser back to. The state must name an open consent, the
    browser must hold the cookie the handoff set, and the consent is used once. The router then
    exchanges the code and sends the browser to the dashboard with connection_id and status: connected,
    denied, failed, or account_mismatch when a reconnect came back with another provider account and the
    old grant was kept. Unauthenticated because the browser arrives from the provider.

    Args:
        state (str | Unset):
        code (str | Unset):
        iss (str | Unset):
        error (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        state=state,
        code=code,
        iss=iss,
        error=error,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    state: str | Unset = UNSET,
    code: str | Unset = UNSET,
    iss: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Any | ErrorResponse | None:
    """Finish a consent

     The redirect URI a provider sends the browser back to. The state must name an open consent, the
    browser must hold the cookie the handoff set, and the consent is used once. The router then
    exchanges the code and sends the browser to the dashboard with connection_id and status: connected,
    denied, failed, or account_mismatch when a reconnect came back with another provider account and the
    old grant was kept. Unauthenticated because the browser arrives from the provider.

    Args:
        state (str | Unset):
        code (str | Unset):
        iss (str | Unset):
        error (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            state=state,
            code=code,
            iss=iss,
            error=error,
        )
    ).parsed
