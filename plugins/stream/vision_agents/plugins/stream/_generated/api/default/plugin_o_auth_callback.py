from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    code: str | Unset = UNSET,
    state: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["code"] = code

    params["state"] = state

    params["error"] = error

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/plugins/callback",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | None:
    if response.status_code == 302:
        return None

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    code: str | Unset = UNSET,
    state: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Response[Any]:
    """Finish a plugin login

     The provider redirects here with a code. The path is unauthenticated because the browser arrives
    from the identity provider, and the state is the secret.

    Args:
        code (str | Unset):
        state (str | Unset):
        error (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        code=code,
        state=state,
        error=error,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    code: str | Unset = UNSET,
    state: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Response[Any]:
    """Finish a plugin login

     The provider redirects here with a code. The path is unauthenticated because the browser arrives
    from the identity provider, and the state is the secret.

    Args:
        code (str | Unset):
        state (str | Unset):
        error (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any]
    """

    kwargs = _get_kwargs(
        code=code,
        state=state,
        error=error,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)
