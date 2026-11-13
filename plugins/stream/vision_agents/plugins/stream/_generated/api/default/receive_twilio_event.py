from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/phone/hooks/twilio/events",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

    if response.status_code == 400:
        response_400 = cast(Any, None)
        return response_400

    if response.status_code == 401:
        response_401 = cast(Any, None)
        return response_401

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if response.status_code == 413:
        response_413 = cast(Any, None)
        return response_413

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
) -> Response[Any | ErrorResponse]:
    """Receive a Twilio account event

     Where the Twilio account sends its Monitor alerts (the Debugger webhook) and its usage triggers'
    callbacks, as form posts. Unauthenticated because Twilio is not a customer: each event is checked
    against X-Twilio-Signature, made with the account's Auth Token over ROUTER_PUBLIC_URL and this path,
    then logged and counted in vendor_account_events_total by its error code or usage category and its
    level. An alert whose Sid was counted in the last hour is answered 204 and not counted again. A
    deployment with no Twilio webhook token or no public URL answers 404, so nobody else can raise the
    count.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Receive a Twilio account event

     Where the Twilio account sends its Monitor alerts (the Debugger webhook) and its usage triggers'
    callbacks, as form posts. Unauthenticated because Twilio is not a customer: each event is checked
    against X-Twilio-Signature, made with the account's Auth Token over ROUTER_PUBLIC_URL and this path,
    then logged and counted in vendor_account_events_total by its error code or usage category and its
    level. An alert whose Sid was counted in the last hour is answered 204 and not counted again. A
    deployment with no Twilio webhook token or no public URL answers 404, so nobody else can raise the
    count.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Receive a Twilio account event

     Where the Twilio account sends its Monitor alerts (the Debugger webhook) and its usage triggers'
    callbacks, as form posts. Unauthenticated because Twilio is not a customer: each event is checked
    against X-Twilio-Signature, made with the account's Auth Token over ROUTER_PUBLIC_URL and this path,
    then logged and counted in vendor_account_events_total by its error code or usage category and its
    level. An alert whose Sid was counted in the last hour is answered 204 and not counted again. A
    deployment with no Twilio webhook token or no public URL answers 404, so nobody else can raise the
    count.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Receive a Twilio account event

     Where the Twilio account sends its Monitor alerts (the Debugger webhook) and its usage triggers'
    callbacks, as form posts. Unauthenticated because Twilio is not a customer: each event is checked
    against X-Twilio-Signature, made with the account's Auth Token over ROUTER_PUBLIC_URL and this path,
    then logged and counted in vendor_account_events_total by its error code or usage category and its
    level. An alert whose Sid was counted in the last hour is answered 204 and not counted again. A
    deployment with no Twilio webhook token or no public URL answers 404, so nobody else can raise the
    count.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
