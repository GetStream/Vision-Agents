from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    force: bool | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["force"] = force

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/agents/connectors/{id}".format(
            id=quote(str(id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

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
    id: str,
    *,
    client: AuthenticatedClient | Client,
    force: bool | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
    """Delete a custom connector

     Deletes one of the app's own connectors, every revision of it, with the app's OAuth client for it. A
    built-in cannot be deleted and is not found. A connector a live connection was made from, or an
    agent config binds, is refused with a 409 naming them, unless force is set: then its connections are
    deleted as a forced connection delete deletes one, credentials dropped at once, and the bindings are
    left in place. The same id may be created again, from revision 1.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The app's custom connector, such as custom_crm.
        force (bool | Unset): Delete it even while connections or agent config bindings use it.
            Its connections are deleted with it, and the bindings are left in place, naming a
            connector that no longer exists.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        force=force,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    force: bool | Unset = UNSET,
) -> Any | ErrorResponse | None:
    """Delete a custom connector

     Deletes one of the app's own connectors, every revision of it, with the app's OAuth client for it. A
    built-in cannot be deleted and is not found. A connector a live connection was made from, or an
    agent config binds, is refused with a 409 naming them, unless force is set: then its connections are
    deleted as a forced connection delete deletes one, credentials dropped at once, and the bindings are
    left in place. The same id may be created again, from revision 1.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The app's custom connector, such as custom_crm.
        force (bool | Unset): Delete it even while connections or agent config bindings use it.
            Its connections are deleted with it, and the bindings are left in place, naming a
            connector that no longer exists.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
        force=force,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    force: bool | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
    """Delete a custom connector

     Deletes one of the app's own connectors, every revision of it, with the app's OAuth client for it. A
    built-in cannot be deleted and is not found. A connector a live connection was made from, or an
    agent config binds, is refused with a 409 naming them, unless force is set: then its connections are
    deleted as a forced connection delete deletes one, credentials dropped at once, and the bindings are
    left in place. The same id may be created again, from revision 1.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The app's custom connector, such as custom_crm.
        force (bool | Unset): Delete it even while connections or agent config bindings use it.
            Its connections are deleted with it, and the bindings are left in place, naming a
            connector that no longer exists.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        force=force,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    force: bool | Unset = UNSET,
) -> Any | ErrorResponse | None:
    """Delete a custom connector

     Deletes one of the app's own connectors, every revision of it, with the app's OAuth client for it. A
    built-in cannot be deleted and is not found. A connector a live connection was made from, or an
    agent config binds, is refused with a 409 naming them, unless force is set: then its connections are
    deleted as a forced connection delete deletes one, credentials dropped at once, and the bindings are
    left in place. The same id may be created again, from revision 1.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The app's custom connector, such as custom_crm.
        force (bool | Unset): Delete it even while connections or agent config bindings use it.
            Its connections are deleted with it, and the bindings are left in place, naming a
            connector that no longer exists.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            force=force,
        )
    ).parsed
