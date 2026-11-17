from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.contact_page import ContactPage
from ...models.contact_stage import ContactStage
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    stage: ContactStage | Unset = UNSET,
    number: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["limit"] = limit

    params["cursor"] = cursor

    json_stage: str | Unset = UNSET
    if not isinstance(stage, Unset):
        json_stage = stage.value

    params["stage"] = json_stage

    params["number"] = number

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/campaigns/{id}/contacts".format(
            id=quote(str(id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ContactPage | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ContactPage.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ContactPage | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    stage: ContactStage | Unset = UNSET,
    number: str | Unset = UNSET,
) -> Response[ContactPage | ErrorResponse]:
    """Who a campaign rings, and how far it has got

     In the order they are rung.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.
        stage (ContactStage | Unset): Where a contact is: enrolled while waiting to be rung,
            in_progress while a call to them happens or is being read, completed once the sequence is
            done with them, rejected when their number was refused before any call, stopped when they
            were left unrung, failed when a call could not be made.
        number (str | Unset): Only the contacts whose number contains this.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContactPage | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        limit=limit,
        cursor=cursor,
        stage=stage,
        number=number,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    stage: ContactStage | Unset = UNSET,
    number: str | Unset = UNSET,
) -> ContactPage | ErrorResponse | None:
    """Who a campaign rings, and how far it has got

     In the order they are rung.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.
        stage (ContactStage | Unset): Where a contact is: enrolled while waiting to be rung,
            in_progress while a call to them happens or is being read, completed once the sequence is
            done with them, rejected when their number was refused before any call, stopped when they
            were left unrung, failed when a call could not be made.
        number (str | Unset): Only the contacts whose number contains this.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContactPage | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
        limit=limit,
        cursor=cursor,
        stage=stage,
        number=number,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    stage: ContactStage | Unset = UNSET,
    number: str | Unset = UNSET,
) -> Response[ContactPage | ErrorResponse]:
    """Who a campaign rings, and how far it has got

     In the order they are rung.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.
        stage (ContactStage | Unset): Where a contact is: enrolled while waiting to be rung,
            in_progress while a call to them happens or is being read, completed once the sequence is
            done with them, rejected when their number was refused before any call, stopped when they
            were left unrung, failed when a call could not be made.
        number (str | Unset): Only the contacts whose number contains this.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContactPage | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        limit=limit,
        cursor=cursor,
        stage=stage,
        number=number,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    stage: ContactStage | Unset = UNSET,
    number: str | Unset = UNSET,
) -> ContactPage | ErrorResponse | None:
    """Who a campaign rings, and how far it has got

     In the order they are rung.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The resource, as returned when it was created.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.
        stage (ContactStage | Unset): Where a contact is: enrolled while waiting to be rung,
            in_progress while a call to them happens or is being read, completed once the sequence is
            done with them, rejected when their number was refused before any call, stopped when they
            were left unrung, failed when a call could not be made.
        number (str | Unset): Only the contacts whose number contains this.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContactPage | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            limit=limit,
            cursor=cursor,
            stage=stage,
            number=number,
        )
    ).parsed
