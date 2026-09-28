from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.policy import Policy
from ...types import Response


def _get_kwargs(
    *,
    body: Policy,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/policies/organization",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | Policy | None:
    if response.status_code == 200:
        response_200 = Policy.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = Error.from_dict(response.json())

        return response_403

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | Policy]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: Policy,
) -> Response[Error | Policy]:
    """Replace the calling app's organization's policy

     Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (Policy): What an organization or an app decided about spend, data handling and
            prompt injection. Every field is optional, and a field left out is no opinion rather than
            off.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Policy]
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
    body: Policy,
) -> Error | Policy | None:
    """Replace the calling app's organization's policy

     Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (Policy): What an organization or an app decided about spend, data handling and
            prompt injection. Every field is optional, and a field left out is no opinion rather than
            off.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Policy
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: Policy,
) -> Response[Error | Policy]:
    """Replace the calling app's organization's policy

     Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (Policy): What an organization or an app decided about spend, data handling and
            prompt injection. Every field is optional, and a field left out is no opinion rather than
            off.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Policy]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: Policy,
) -> Error | Policy | None:
    """Replace the calling app's organization's policy

     Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (Policy): What an organization or an app decided about spend, data handling and
            prompt injection. Every field is optional, and a field left out is no opinion rather than
            off.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Policy
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
