from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_opt_out_request import CreateOptOutRequest
from ...models.error import Error
from ...models.opt_out import OptOut
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: CreateOptOutRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/phone/opt-outs",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | OptOut | None:
    if response.status_code == 201:
        response_201 = OptOut.from_dict(response.json())

        return response_201

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = Error.from_dict(response.json())

        return response_403

    if response.status_code == 500:
        response_500 = Error.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | OptOut]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CreateOptOutRequest | Unset = UNSET,
) -> Response[Error | OptOut]:
    """Record an opt-out

     Stops every text and call to a recipient on a channel, or on all of them. Somebody texting STOP is
    recorded without this.

    Server-side only.

    Args:
        body (CreateOptOutRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | OptOut]
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
    body: CreateOptOutRequest | Unset = UNSET,
) -> Error | OptOut | None:
    """Record an opt-out

     Stops every text and call to a recipient on a channel, or on all of them. Somebody texting STOP is
    recorded without this.

    Server-side only.

    Args:
        body (CreateOptOutRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | OptOut
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CreateOptOutRequest | Unset = UNSET,
) -> Response[Error | OptOut]:
    """Record an opt-out

     Stops every text and call to a recipient on a channel, or on all of them. Somebody texting STOP is
    recorded without this.

    Server-side only.

    Args:
        body (CreateOptOutRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | OptOut]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: CreateOptOutRequest | Unset = UNSET,
) -> Error | OptOut | None:
    """Record an opt-out

     Stops every text and call to a recipient on a channel, or on all of them. Somebody texting STOP is
    recorded without this.

    Server-side only.

    Args:
        body (CreateOptOutRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | OptOut
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
