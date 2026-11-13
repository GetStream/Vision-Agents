from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.tool_approval_answer import ToolApprovalAnswer
from ...types import Response


def _get_kwargs(
    cid: str,
    *,
    body: ToolApprovalAnswer,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/conversations/{cid}/tool_approvals".format(
            cid=quote(str(cid), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
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
) -> Response[Any | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolApprovalAnswer,
) -> Response[Any | ErrorResponse]:
    """Answer a call waiting for a person's approval

     Allows or declines a connector call whose binding sets `require_approval`, held on its
    `ai_tool_call` step (`awaiting_approval`) in the conversation's channel, as `tool_approval` on the
    session's socket does. The step's `target_user_id` is who may answer: the session's user, or in a
    channel of a type an agent config answers in (`chat_channels`), the person whose message the reply
    answers, with their own user token while they are a member of the channel. The app's backend may
    answer any. Allowed, the call is sent; declined, it is not, and the model is told so.
    A call held for another person to allow is a 403. A call nobody waits on, answered or stopped
    already, or in a channel the caller is not a member of, is a 404.

    Args:
        cid (str): The conversation's channel, type:id.
        body (ToolApprovalAnswer): A person's answer to a connector call waiting for their
            approval. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        cid=cid,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolApprovalAnswer,
) -> Any | ErrorResponse | None:
    """Answer a call waiting for a person's approval

     Allows or declines a connector call whose binding sets `require_approval`, held on its
    `ai_tool_call` step (`awaiting_approval`) in the conversation's channel, as `tool_approval` on the
    session's socket does. The step's `target_user_id` is who may answer: the session's user, or in a
    channel of a type an agent config answers in (`chat_channels`), the person whose message the reply
    answers, with their own user token while they are a member of the channel. The app's backend may
    answer any. Allowed, the call is sent; declined, it is not, and the model is told so.
    A call held for another person to allow is a 403. A call nobody waits on, answered or stopped
    already, or in a channel the caller is not a member of, is a 404.

    Args:
        cid (str): The conversation's channel, type:id.
        body (ToolApprovalAnswer): A person's answer to a connector call waiting for their
            approval. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        cid=cid,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolApprovalAnswer,
) -> Response[Any | ErrorResponse]:
    """Answer a call waiting for a person's approval

     Allows or declines a connector call whose binding sets `require_approval`, held on its
    `ai_tool_call` step (`awaiting_approval`) in the conversation's channel, as `tool_approval` on the
    session's socket does. The step's `target_user_id` is who may answer: the session's user, or in a
    channel of a type an agent config answers in (`chat_channels`), the person whose message the reply
    answers, with their own user token while they are a member of the channel. The app's backend may
    answer any. Allowed, the call is sent; declined, it is not, and the model is told so.
    A call held for another person to allow is a 403. A call nobody waits on, answered or stopped
    already, or in a channel the caller is not a member of, is a 404.

    Args:
        cid (str): The conversation's channel, type:id.
        body (ToolApprovalAnswer): A person's answer to a connector call waiting for their
            approval. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        cid=cid,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    cid: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolApprovalAnswer,
) -> Any | ErrorResponse | None:
    """Answer a call waiting for a person's approval

     Allows or declines a connector call whose binding sets `require_approval`, held on its
    `ai_tool_call` step (`awaiting_approval`) in the conversation's channel, as `tool_approval` on the
    session's socket does. The step's `target_user_id` is who may answer: the session's user, or in a
    channel of a type an agent config answers in (`chat_channels`), the person whose message the reply
    answers, with their own user token while they are a member of the channel. The app's backend may
    answer any. Allowed, the call is sent; declined, it is not, and the model is told so.
    A call held for another person to allow is a 403. A call nobody waits on, answered or stopped
    already, or in a channel the caller is not a member of, is a 404.

    Args:
        cid (str): The conversation's channel, type:id.
        body (ToolApprovalAnswer): A person's answer to a connector call waiting for their
            approval. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            cid=cid,
            client=client,
            body=body,
        )
    ).parsed
