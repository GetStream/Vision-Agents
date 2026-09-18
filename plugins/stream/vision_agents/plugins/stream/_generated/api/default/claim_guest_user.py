from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.claim_guest_request import ClaimGuestRequest
from ...models.claim_guest_result import ClaimGuestResult
from ...models.error import Error
from ...types import Response


def _get_kwargs(
    *,
    body: ClaimGuestRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/guests/claim",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ClaimGuestResult | Error | None:
    if response.status_code == 200:
        response_200 = ClaimGuestResult.from_dict(response.json())

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

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if response.status_code == 409:
        response_409 = Error.from_dict(response.json())

        return response_409

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ClaimGuestResult | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ClaimGuestRequest,
) -> Response[ClaimGuestResult | Error]:
    """Move a guest's conversations onto the account they turned out to be

     For somebody who talked to an agent and then signed up. Their sessions are rewritten to the real
    user and the guest is marked claimed, in one transaction: a guest marked claimed whose sessions
    still say the guest owns them is a person who signed up and lost their history, and sessions moved
    without the guest being marked is a guest that can be claimed again, by somebody else.
    Server-side only, and the one operation here that most needs to be. Only the customer's own backend
    knows that a given guest is a given account -- it is the thing that just authenticated them. A page
    allowed to ask this could claim anybody's conversations by guessing a guest id, which is the whole
    attack.
    A guest already claimed is refused rather than moved again.

    Args:
        body (ClaimGuestRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ClaimGuestResult | Error]
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
    body: ClaimGuestRequest,
) -> ClaimGuestResult | Error | None:
    """Move a guest's conversations onto the account they turned out to be

     For somebody who talked to an agent and then signed up. Their sessions are rewritten to the real
    user and the guest is marked claimed, in one transaction: a guest marked claimed whose sessions
    still say the guest owns them is a person who signed up and lost their history, and sessions moved
    without the guest being marked is a guest that can be claimed again, by somebody else.
    Server-side only, and the one operation here that most needs to be. Only the customer's own backend
    knows that a given guest is a given account -- it is the thing that just authenticated them. A page
    allowed to ask this could claim anybody's conversations by guessing a guest id, which is the whole
    attack.
    A guest already claimed is refused rather than moved again.

    Args:
        body (ClaimGuestRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ClaimGuestResult | Error
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ClaimGuestRequest,
) -> Response[ClaimGuestResult | Error]:
    """Move a guest's conversations onto the account they turned out to be

     For somebody who talked to an agent and then signed up. Their sessions are rewritten to the real
    user and the guest is marked claimed, in one transaction: a guest marked claimed whose sessions
    still say the guest owns them is a person who signed up and lost their history, and sessions moved
    without the guest being marked is a guest that can be claimed again, by somebody else.
    Server-side only, and the one operation here that most needs to be. Only the customer's own backend
    knows that a given guest is a given account -- it is the thing that just authenticated them. A page
    allowed to ask this could claim anybody's conversations by guessing a guest id, which is the whole
    attack.
    A guest already claimed is refused rather than moved again.

    Args:
        body (ClaimGuestRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ClaimGuestResult | Error]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ClaimGuestRequest,
) -> ClaimGuestResult | Error | None:
    """Move a guest's conversations onto the account they turned out to be

     For somebody who talked to an agent and then signed up. Their sessions are rewritten to the real
    user and the guest is marked claimed, in one transaction: a guest marked claimed whose sessions
    still say the guest owns them is a person who signed up and lost their history, and sessions moved
    without the guest being marked is a guest that can be claimed again, by somebody else.
    Server-side only, and the one operation here that most needs to be. Only the customer's own backend
    knows that a given guest is a given account -- it is the thing that just authenticated them. A page
    allowed to ask this could claim anybody's conversations by guessing a guest id, which is the whole
    attack.
    A guest already claimed is refused rather than moved again.

    Args:
        body (ClaimGuestRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ClaimGuestResult | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
