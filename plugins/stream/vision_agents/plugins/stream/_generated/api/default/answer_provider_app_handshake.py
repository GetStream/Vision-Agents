from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connector_id: str,
    provider_app_id: str,
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
        "url": "/v1/connectors/events/{connector_id}/{provider_app_id}".format(
            connector_id=quote(str(connector_id), safe=""),
            provider_app_id=quote(str(provider_app_id), safe=""),
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

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if response.status_code == 405:
        response_405 = cast(Any, None)
        return response_405

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
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Response[Any | ErrorResponse | str]:
    """Answer a provider app's handshake

     Where a provider checks a provider app's events URL before it delivers to it: Meta's Verify Token
    check of a customer's WhatsApp webhook, for one. Unauthenticated because the provider is not a
    customer. Only a connector whose manifest declares channel.handshake answers it; the verify token is
    the provider app's id, the one in the URL, so nothing is stored for it, and every delivery is still
    verified with the app's own secret. With hub.mode subscribe, hub.verify_token the provider app's id
    and hub.challenge digits only, the challenge is echoed as text/plain. Any other connector, an
    unknown provider app, or a deployment without connectors answers 405 as for any method a route does
    not serve. No SDK wraps it: only a provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):
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
        connector_id=connector_id,
        provider_app_id=provider_app_id,
        hub_mode=hub_mode,
        hub_verify_token=hub_verify_token,
        hub_challenge=hub_challenge,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Any | ErrorResponse | str | None:
    """Answer a provider app's handshake

     Where a provider checks a provider app's events URL before it delivers to it: Meta's Verify Token
    check of a customer's WhatsApp webhook, for one. Unauthenticated because the provider is not a
    customer. Only a connector whose manifest declares channel.handshake answers it; the verify token is
    the provider app's id, the one in the URL, so nothing is stored for it, and every delivery is still
    verified with the app's own secret. With hub.mode subscribe, hub.verify_token the provider app's id
    and hub.challenge digits only, the challenge is echoed as text/plain. Any other connector, an
    unknown provider app, or a deployment without connectors answers 405 as for any method a route does
    not serve. No SDK wraps it: only a provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):
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
        connector_id=connector_id,
        provider_app_id=provider_app_id,
        client=client,
        hub_mode=hub_mode,
        hub_verify_token=hub_verify_token,
        hub_challenge=hub_challenge,
    ).parsed


async def asyncio_detailed(
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Response[Any | ErrorResponse | str]:
    """Answer a provider app's handshake

     Where a provider checks a provider app's events URL before it delivers to it: Meta's Verify Token
    check of a customer's WhatsApp webhook, for one. Unauthenticated because the provider is not a
    customer. Only a connector whose manifest declares channel.handshake answers it; the verify token is
    the provider app's id, the one in the URL, so nothing is stored for it, and every delivery is still
    verified with the app's own secret. With hub.mode subscribe, hub.verify_token the provider app's id
    and hub.challenge digits only, the challenge is echoed as text/plain. Any other connector, an
    unknown provider app, or a deployment without connectors answers 405 as for any method a route does
    not serve. No SDK wraps it: only a provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):
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
        connector_id=connector_id,
        provider_app_id=provider_app_id,
        hub_mode=hub_mode,
        hub_verify_token=hub_verify_token,
        hub_challenge=hub_challenge,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
    hub_mode: str | Unset = UNSET,
    hub_verify_token: str | Unset = UNSET,
    hub_challenge: str | Unset = UNSET,
) -> Any | ErrorResponse | str | None:
    """Answer a provider app's handshake

     Where a provider checks a provider app's events URL before it delivers to it: Meta's Verify Token
    check of a customer's WhatsApp webhook, for one. Unauthenticated because the provider is not a
    customer. Only a connector whose manifest declares channel.handshake answers it; the verify token is
    the provider app's id, the one in the URL, so nothing is stored for it, and every delivery is still
    verified with the app's own secret. With hub.mode subscribe, hub.verify_token the provider app's id
    and hub.challenge digits only, the challenge is echoed as text/plain. Any other connector, an
    unknown provider app, or a deployment without connectors answers 405 as for any method a route does
    not serve. No SDK wraps it: only a provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):
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
            connector_id=connector_id,
            provider_app_id=provider_app_id,
            client=client,
            hub_mode=hub_mode,
            hub_verify_token=hub_verify_token,
            hub_challenge=hub_challenge,
        )
    ).parsed
