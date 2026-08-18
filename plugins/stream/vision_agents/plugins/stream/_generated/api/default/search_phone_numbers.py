from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.available_number import AvailableNumber
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    vendor: str,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    limit: int | Unset = 10,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["vendor"] = vendor

    params["country"] = country

    params["area_code"] = area_code

    params["contains"] = contains

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/phone/numbers/available",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[AvailableNumber] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = AvailableNumber.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

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
) -> Response[Error | list[AvailableNumber]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    vendor: str,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    limit: int | Unset = 10,
) -> Response[Error | list[AvailableNumber]]:
    """Search a vendor for numbers to buy

    Args:
        vendor (str):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[AvailableNumber]]
    """

    kwargs = _get_kwargs(
        vendor=vendor,
        country=country,
        area_code=area_code,
        contains=contains,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    vendor: str,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    limit: int | Unset = 10,
) -> Error | list[AvailableNumber] | None:
    """Search a vendor for numbers to buy

    Args:
        vendor (str):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[AvailableNumber]
    """

    return sync_detailed(
        client=client,
        vendor=vendor,
        country=country,
        area_code=area_code,
        contains=contains,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    vendor: str,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    limit: int | Unset = 10,
) -> Response[Error | list[AvailableNumber]]:
    """Search a vendor for numbers to buy

    Args:
        vendor (str):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[AvailableNumber]]
    """

    kwargs = _get_kwargs(
        vendor=vendor,
        country=country,
        area_code=area_code,
        contains=contains,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    vendor: str,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    limit: int | Unset = 10,
) -> Error | list[AvailableNumber] | None:
    """Search a vendor for numbers to buy

    Args:
        vendor (str):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[AvailableNumber]
    """

    return (
        await asyncio_detailed(
            client=client,
            vendor=vendor,
            country=country,
            area_code=area_code,
            contains=contains,
            limit=limit,
        )
    ).parsed
