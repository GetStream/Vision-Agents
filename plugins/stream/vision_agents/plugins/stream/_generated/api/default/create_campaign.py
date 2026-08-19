from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.campaign import Campaign
from ...models.campaign_request import CampaignRequest
from ...models.error import Error
from ...types import Response


def _get_kwargs(
    *,
    body: CampaignRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/campaigns",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Campaign | Error | None:
    if response.status_code == 201:
        response_201 = Campaign.from_dict(response.json())

        return response_201

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Campaign | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CampaignRequest,
) -> Response[Campaign | Error]:
    """Define a list of people to ring

     A campaign is created stopped. Add the people to ring, then start it: concurrency is how many of
    these calls may be happening at once, and every call is a conversation this service is running.

    Args:
        body (CampaignRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Campaign | Error]
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
    body: CampaignRequest,
) -> Campaign | Error | None:
    """Define a list of people to ring

     A campaign is created stopped. Add the people to ring, then start it: concurrency is how many of
    these calls may be happening at once, and every call is a conversation this service is running.

    Args:
        body (CampaignRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Campaign | Error
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CampaignRequest,
) -> Response[Campaign | Error]:
    """Define a list of people to ring

     A campaign is created stopped. Add the people to ring, then start it: concurrency is how many of
    these calls may be happening at once, and every call is a conversation this service is running.

    Args:
        body (CampaignRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Campaign | Error]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: CampaignRequest,
) -> Campaign | Error | None:
    """Define a list of people to ring

     A campaign is created stopped. Add the people to ring, then start it: concurrency is how many of
    these calls may be happening at once, and every call is a conversation this service is running.

    Args:
        body (CampaignRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Campaign | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
