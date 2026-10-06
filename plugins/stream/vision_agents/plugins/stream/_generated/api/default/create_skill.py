from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.skill import Skill
from ...models.skill_request import SkillRequest
from ...types import Response


def _get_kwargs(
    *,
    body: SkillRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/skills",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | Skill | None:
    if response.status_code == 201:
        response_201 = Skill.from_dict(response.json())

        return response_201

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

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
) -> Response[ErrorResponse | Skill]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SkillRequest,
) -> Response[ErrorResponse | Skill]:
    """Define a kind of work worth handing to the slower model

     A skill belongs to one agent config. Two agents that both need the same kind of work have one each,
    so editing what "explain" means for one leaves the other alone. The built-in think, recall and
    explain need no row: a config may name them without defining them.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SkillRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Skill]
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
    body: SkillRequest,
) -> ErrorResponse | Skill | None:
    """Define a kind of work worth handing to the slower model

     A skill belongs to one agent config. Two agents that both need the same kind of work have one each,
    so editing what "explain" means for one leaves the other alone. The built-in think, recall and
    explain need no row: a config may name them without defining them.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SkillRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Skill
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SkillRequest,
) -> Response[ErrorResponse | Skill]:
    """Define a kind of work worth handing to the slower model

     A skill belongs to one agent config. Two agents that both need the same kind of work have one each,
    so editing what "explain" means for one leaves the other alone. The built-in think, recall and
    explain need no row: a config may name them without defining them.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SkillRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Skill]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: SkillRequest,
) -> ErrorResponse | Skill | None:
    """Define a kind of work worth handing to the slower model

     A skill belongs to one agent config. Two agents that both need the same kind of work have one each,
    so editing what "explain" means for one leaves the other alone. The built-in think, recall and
    explain need no row: a config may name them without defining them.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SkillRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Skill
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
