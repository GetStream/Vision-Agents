from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.use_case import UseCase
from ...models.use_case_request import UseCaseRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: UseCaseRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/phone/use-cases",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | UseCase | None:
    if response.status_code == 201:
        response_201 = UseCase.from_dict(response.json())

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
) -> Response[Error | UseCase]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: UseCaseRequest | Unset = UNSET,
) -> Response[Error | UseCase]:
    """Create a 10DLC use case

     Saves a draft use case. Nothing is checked beyond its shape until it is submitted.

    Server-side only.

    Args:
        body (UseCaseRequest | Unset): What an app sends texts and makes calls for. Saved as a
            draft and submitted for review once it and the business profile are complete.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | UseCase]
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
    body: UseCaseRequest | Unset = UNSET,
) -> Error | UseCase | None:
    """Create a 10DLC use case

     Saves a draft use case. Nothing is checked beyond its shape until it is submitted.

    Server-side only.

    Args:
        body (UseCaseRequest | Unset): What an app sends texts and makes calls for. Saved as a
            draft and submitted for review once it and the business profile are complete.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | UseCase
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: UseCaseRequest | Unset = UNSET,
) -> Response[Error | UseCase]:
    """Create a 10DLC use case

     Saves a draft use case. Nothing is checked beyond its shape until it is submitted.

    Server-side only.

    Args:
        body (UseCaseRequest | Unset): What an app sends texts and makes calls for. Saved as a
            draft and submitted for review once it and the business profile are complete.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | UseCase]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: UseCaseRequest | Unset = UNSET,
) -> Error | UseCase | None:
    """Create a 10DLC use case

     Saves a draft use case. Nothing is checked beyond its shape until it is submitted.

    Server-side only.

    Args:
        body (UseCaseRequest | Unset): What an app sends texts and makes calls for. Saved as a
            draft and submitted for review once it and the business profile are complete.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | UseCase
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
