from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_directory import AgentDirectory
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/configs/{id}/directory".format(
            id=quote(str(id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentDirectory | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = AgentDirectory.from_dict(response.json())

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
) -> Response[AgentDirectory | ErrorResponse]:
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
) -> Response[AgentDirectory | ErrorResponse]:
    """An agent config as the files of its directory

     The reverse of a sync: the agent as the directory it would be synced from. The files are agent.yaml,
    instructions.md, guardrail.md, each skill as skills/<name>/SKILL.md with the files beside it, the
    agent's knowledge base as knowledge/<source> with the pages it is kept filled from in
    knowledge/urls.yaml, its simulations in simulations/simulations.yaml, and ui/components.json and
    ui/blueprints.yaml when a directory's files gave the agent its UI. agent.yaml is canonical: its keys
    in a fixed order, what a sync fills in on its own left out (region us, the default harness, a call
    tool at its default, a connector's tools when it grants every one), and a connector that says only
    its name written as the name alone.

    Sent to POST /v1/agents/sync/files, the files store the agent as it is, apart from what
    `not_exported` lists. What a directory never says is not in the files and a sync leaves it alone:
    llm_fallbacks, review, resume_history, visible_tools and episode_cards. A fixed connector is written
    with its connection's id, whatever label agent.yaml first named it by, and a skill the config no
    longer names is left out.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentDirectory | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> AgentDirectory | ErrorResponse | None:
    """An agent config as the files of its directory

     The reverse of a sync: the agent as the directory it would be synced from. The files are agent.yaml,
    instructions.md, guardrail.md, each skill as skills/<name>/SKILL.md with the files beside it, the
    agent's knowledge base as knowledge/<source> with the pages it is kept filled from in
    knowledge/urls.yaml, its simulations in simulations/simulations.yaml, and ui/components.json and
    ui/blueprints.yaml when a directory's files gave the agent its UI. agent.yaml is canonical: its keys
    in a fixed order, what a sync fills in on its own left out (region us, the default harness, a call
    tool at its default, a connector's tools when it grants every one), and a connector that says only
    its name written as the name alone.

    Sent to POST /v1/agents/sync/files, the files store the agent as it is, apart from what
    `not_exported` lists. What a directory never says is not in the files and a sync leaves it alone:
    llm_fallbacks, review, resume_history, visible_tools and episode_cards. A fixed connector is written
    with its connection's id, whatever label agent.yaml first named it by, and a skill the config no
    longer names is left out.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentDirectory | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[AgentDirectory | ErrorResponse]:
    """An agent config as the files of its directory

     The reverse of a sync: the agent as the directory it would be synced from. The files are agent.yaml,
    instructions.md, guardrail.md, each skill as skills/<name>/SKILL.md with the files beside it, the
    agent's knowledge base as knowledge/<source> with the pages it is kept filled from in
    knowledge/urls.yaml, its simulations in simulations/simulations.yaml, and ui/components.json and
    ui/blueprints.yaml when a directory's files gave the agent its UI. agent.yaml is canonical: its keys
    in a fixed order, what a sync fills in on its own left out (region us, the default harness, a call
    tool at its default, a connector's tools when it grants every one), and a connector that says only
    its name written as the name alone.

    Sent to POST /v1/agents/sync/files, the files store the agent as it is, apart from what
    `not_exported` lists. What a directory never says is not in the files and a sync leaves it alone:
    llm_fallbacks, review, resume_history, visible_tools and episode_cards. A fixed connector is written
    with its connection's id, whatever label agent.yaml first named it by, and a skill the config no
    longer names is left out.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentDirectory | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> AgentDirectory | ErrorResponse | None:
    """An agent config as the files of its directory

     The reverse of a sync: the agent as the directory it would be synced from. The files are agent.yaml,
    instructions.md, guardrail.md, each skill as skills/<name>/SKILL.md with the files beside it, the
    agent's knowledge base as knowledge/<source> with the pages it is kept filled from in
    knowledge/urls.yaml, its simulations in simulations/simulations.yaml, and ui/components.json and
    ui/blueprints.yaml when a directory's files gave the agent its UI. agent.yaml is canonical: its keys
    in a fixed order, what a sync fills in on its own left out (region us, the default harness, a call
    tool at its default, a connector's tools when it grants every one), and a connector that says only
    its name written as the name alone.

    Sent to POST /v1/agents/sync/files, the files store the agent as it is, apart from what
    `not_exported` lists. What a directory never says is not in the files and a sync leaves it alone:
    llm_fallbacks, review, resume_history, visible_tools and episode_cards. A fixed connector is written
    with its connection's id, whatever label agent.yaml first named it by, and a skill the config no
    longer names is left out.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentDirectory | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
