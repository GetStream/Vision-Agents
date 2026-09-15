import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_log_page import AgentLogPage
from ...models.error import Error
from ...models.list_agent_logs_severity import ListAgentLogsSeverity
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    severity: ListAgentLogsSeverity | Unset = UNSET,
    source: str | Unset = UNSET,
    q: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    cursor: str | Unset = UNSET,
    limit: int | Unset = 250,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["config_id"] = config_id

    params["session_id"] = session_id

    params["user_id"] = user_id

    json_severity: str | Unset = UNSET
    if not isinstance(severity, Unset):
        json_severity = severity.value

    params["severity"] = json_severity

    params["source"] = source

    params["q"] = q

    json_from_: str | Unset = UNSET
    if not isinstance(from_, Unset):
        json_from_ = from_.isoformat()
    params["from"] = json_from_

    json_to: str | Unset = UNSET
    if not isinstance(to, Unset):
        json_to = to.isoformat()
    params["to"] = json_to

    params["cursor"] = cursor

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/logs",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentLogPage | Error | None:
    if response.status_code == 200:
        response_200 = AgentLogPage.from_dict(response.json())

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

    if response.status_code == 503:
        response_503 = Error.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[AgentLogPage | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    severity: ListAgentLogsSeverity | Unset = UNSET,
    source: str | Unset = UNSET,
    q: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    cursor: str | Unset = UNSET,
    limit: int | Unset = 250,
) -> Response[AgentLogPage | Error]:
    """Latest structured agent logs, with backward cursor pagination

    Args:
        config_id (str | Unset):
        session_id (str | Unset):
        user_id (str | Unset):
        severity (ListAgentLogsSeverity | Unset):
        source (str | Unset):
        q (str | Unset):
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        cursor (str | Unset):
        limit (int | Unset):  Default: 250.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentLogPage | Error]
    """

    kwargs = _get_kwargs(
        config_id=config_id,
        session_id=session_id,
        user_id=user_id,
        severity=severity,
        source=source,
        q=q,
        from_=from_,
        to=to,
        cursor=cursor,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    severity: ListAgentLogsSeverity | Unset = UNSET,
    source: str | Unset = UNSET,
    q: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    cursor: str | Unset = UNSET,
    limit: int | Unset = 250,
) -> AgentLogPage | Error | None:
    """Latest structured agent logs, with backward cursor pagination

    Args:
        config_id (str | Unset):
        session_id (str | Unset):
        user_id (str | Unset):
        severity (ListAgentLogsSeverity | Unset):
        source (str | Unset):
        q (str | Unset):
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        cursor (str | Unset):
        limit (int | Unset):  Default: 250.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentLogPage | Error
    """

    return sync_detailed(
        client=client,
        config_id=config_id,
        session_id=session_id,
        user_id=user_id,
        severity=severity,
        source=source,
        q=q,
        from_=from_,
        to=to,
        cursor=cursor,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    severity: ListAgentLogsSeverity | Unset = UNSET,
    source: str | Unset = UNSET,
    q: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    cursor: str | Unset = UNSET,
    limit: int | Unset = 250,
) -> Response[AgentLogPage | Error]:
    """Latest structured agent logs, with backward cursor pagination

    Args:
        config_id (str | Unset):
        session_id (str | Unset):
        user_id (str | Unset):
        severity (ListAgentLogsSeverity | Unset):
        source (str | Unset):
        q (str | Unset):
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        cursor (str | Unset):
        limit (int | Unset):  Default: 250.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentLogPage | Error]
    """

    kwargs = _get_kwargs(
        config_id=config_id,
        session_id=session_id,
        user_id=user_id,
        severity=severity,
        source=source,
        q=q,
        from_=from_,
        to=to,
        cursor=cursor,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    severity: ListAgentLogsSeverity | Unset = UNSET,
    source: str | Unset = UNSET,
    q: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    cursor: str | Unset = UNSET,
    limit: int | Unset = 250,
) -> AgentLogPage | Error | None:
    """Latest structured agent logs, with backward cursor pagination

    Args:
        config_id (str | Unset):
        session_id (str | Unset):
        user_id (str | Unset):
        severity (ListAgentLogsSeverity | Unset):
        source (str | Unset):
        q (str | Unset):
        from_ (datetime.datetime | Unset):
        to (datetime.datetime | Unset):
        cursor (str | Unset):
        limit (int | Unset):  Default: 250.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentLogPage | Error
    """

    return (
        await asyncio_detailed(
            client=client,
            config_id=config_id,
            session_id=session_id,
            user_id=user_id,
            severity=severity,
            source=source,
            q=q,
            from_=from_,
            to=to,
            cursor=cursor,
            limit=limit,
        )
    ).parsed
