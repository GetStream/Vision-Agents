from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.place_call_request import PlaceCallRequest
from ...models.placed_call import PlacedCall
from ...types import Response


def _get_kwargs(
    *,
    body: PlaceCallRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/phone/calls",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | PlacedCall | None:
    if response.status_code == 202:
        response_202 = PlacedCall.from_dict(response.json())

        return response_202

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
) -> Response[Error | PlacedCall]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: PlaceCallRequest,
) -> Response[Error | PlacedCall]:
    """Place an outbound call and bridge it into a Stream call

     Stream's SIP is inbound only, so the vendor originates the call and connects it to a trunk the agent
    is already on, rather than Stream dialling out.

    Args:
        body (PlaceCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | PlacedCall]
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
    body: PlaceCallRequest,
) -> Error | PlacedCall | None:
    """Place an outbound call and bridge it into a Stream call

     Stream's SIP is inbound only, so the vendor originates the call and connects it to a trunk the agent
    is already on, rather than Stream dialling out.

    Args:
        body (PlaceCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | PlacedCall
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: PlaceCallRequest,
) -> Response[Error | PlacedCall]:
    """Place an outbound call and bridge it into a Stream call

     Stream's SIP is inbound only, so the vendor originates the call and connects it to a trunk the agent
    is already on, rather than Stream dialling out.

    Args:
        body (PlaceCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | PlacedCall]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: PlaceCallRequest,
) -> Error | PlacedCall | None:
    """Place an outbound call and bridge it into a Stream call

     Stream's SIP is inbound only, so the vendor originates the call and connects it to a trunk the agent
    is already on, rather than Stream dialling out.

    Args:
        body (PlaceCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | PlacedCall
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
