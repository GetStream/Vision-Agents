from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
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
) -> Any | ErrorResponse | None:
    if response.status_code == 302:
        response_302 = cast(Any, None)
        return response_302

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
    code: str | Unset = UNSET,
    state: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
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
        Response[Any | ErrorResponse]
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


def sync(
    *,
    client: AuthenticatedClient | Client,
    code: str | Unset = UNSET,
    state: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Any | ErrorResponse | None:
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
        Any | ErrorResponse
    """

    return sync_detailed(
        client=client,
        code=code,
        state=state,
        error=error,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    code: str | Unset = UNSET,
    state: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
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
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        code=code,
        state=state,
        error=error,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    code: str | Unset = UNSET,
    state: str | Unset = UNSET,
    error: str | Unset = UNSET,
) -> Any | ErrorResponse | None:
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
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            code=code,
            state=state,
            error=error,
        )
    ).parsed
