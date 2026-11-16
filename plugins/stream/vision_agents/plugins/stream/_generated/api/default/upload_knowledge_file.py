from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.indexed_knowledge_document import IndexedKnowledgeDocument
from ...models.upload_knowledge_file_body import UploadKnowledgeFileBody
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: UploadKnowledgeFileBody,
    namespace: str,
    source: str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    params: dict[str, Any] = {}

    params["namespace"] = namespace

    params["source"] = source

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/knowledge/files",
        "params": params,
    }

    _kwargs["files"] = body.to_multipart()

    headers["Content-Type"] = "multipart/form-data; boundary=+++"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | IndexedKnowledgeDocument | None:
    if response.status_code == 202:
        response_202 = IndexedKnowledgeDocument.from_dict(response.json())

        return response_202

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

    if response.status_code == 413:
        response_413 = ErrorResponse.from_dict(response.json())

        return response_413

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
) -> Response[ErrorResponse | IndexedKnowledgeDocument]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: UploadKnowledgeFileBody,
    namespace: str,
    source: str | Unset = UNSET,
) -> Response[ErrorResponse | IndexedKnowledgeDocument]:
    """Upload a file to a knowledge base, to be read in the background

     A PDF, a Word document (DOCX or DOC), an Excel workbook (XLSX), an RTF or OpenDocument text file, or
    Markdown or plain text, of at most 50 MB, sent as multipart/form-data in a part named file. It is
    read by its bytes, whatever it is called. A file whose first bytes are none of these, such as a
    photo, is refused at once.
    The answer is 202 once the file is kept: the document is pending, and is read in the background into
    Markdown that keeps its headings, lists and tables, each passage saying the page it is from. A page
    that is only a scan is read from its picture through the router's ocr-default, billed per page. Poll
    `getKnowledgeDocument` until its state is indexed, or failed with an error saying why: a password, a
    damaged file, or no text at all.
    The passages are keyed by `source`, the file's name unless it is given, so uploading the same source
    again replaces the document; its old passages are still looked up until the new file has been read.
    The file itself is deleted once read. Server-side only.

    Args:
        namespace (str):
        source (str | Unset):
        body (UploadKnowledgeFileBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | IndexedKnowledgeDocument]
    """

    kwargs = _get_kwargs(
        body=body,
        namespace=namespace,
        source=source,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: UploadKnowledgeFileBody,
    namespace: str,
    source: str | Unset = UNSET,
) -> ErrorResponse | IndexedKnowledgeDocument | None:
    """Upload a file to a knowledge base, to be read in the background

     A PDF, a Word document (DOCX or DOC), an Excel workbook (XLSX), an RTF or OpenDocument text file, or
    Markdown or plain text, of at most 50 MB, sent as multipart/form-data in a part named file. It is
    read by its bytes, whatever it is called. A file whose first bytes are none of these, such as a
    photo, is refused at once.
    The answer is 202 once the file is kept: the document is pending, and is read in the background into
    Markdown that keeps its headings, lists and tables, each passage saying the page it is from. A page
    that is only a scan is read from its picture through the router's ocr-default, billed per page. Poll
    `getKnowledgeDocument` until its state is indexed, or failed with an error saying why: a password, a
    damaged file, or no text at all.
    The passages are keyed by `source`, the file's name unless it is given, so uploading the same source
    again replaces the document; its old passages are still looked up until the new file has been read.
    The file itself is deleted once read. Server-side only.

    Args:
        namespace (str):
        source (str | Unset):
        body (UploadKnowledgeFileBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | IndexedKnowledgeDocument
    """

    return sync_detailed(
        client=client,
        body=body,
        namespace=namespace,
        source=source,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: UploadKnowledgeFileBody,
    namespace: str,
    source: str | Unset = UNSET,
) -> Response[ErrorResponse | IndexedKnowledgeDocument]:
    """Upload a file to a knowledge base, to be read in the background

     A PDF, a Word document (DOCX or DOC), an Excel workbook (XLSX), an RTF or OpenDocument text file, or
    Markdown or plain text, of at most 50 MB, sent as multipart/form-data in a part named file. It is
    read by its bytes, whatever it is called. A file whose first bytes are none of these, such as a
    photo, is refused at once.
    The answer is 202 once the file is kept: the document is pending, and is read in the background into
    Markdown that keeps its headings, lists and tables, each passage saying the page it is from. A page
    that is only a scan is read from its picture through the router's ocr-default, billed per page. Poll
    `getKnowledgeDocument` until its state is indexed, or failed with an error saying why: a password, a
    damaged file, or no text at all.
    The passages are keyed by `source`, the file's name unless it is given, so uploading the same source
    again replaces the document; its old passages are still looked up until the new file has been read.
    The file itself is deleted once read. Server-side only.

    Args:
        namespace (str):
        source (str | Unset):
        body (UploadKnowledgeFileBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | IndexedKnowledgeDocument]
    """

    kwargs = _get_kwargs(
        body=body,
        namespace=namespace,
        source=source,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: UploadKnowledgeFileBody,
    namespace: str,
    source: str | Unset = UNSET,
) -> ErrorResponse | IndexedKnowledgeDocument | None:
    """Upload a file to a knowledge base, to be read in the background

     A PDF, a Word document (DOCX or DOC), an Excel workbook (XLSX), an RTF or OpenDocument text file, or
    Markdown or plain text, of at most 50 MB, sent as multipart/form-data in a part named file. It is
    read by its bytes, whatever it is called. A file whose first bytes are none of these, such as a
    photo, is refused at once.
    The answer is 202 once the file is kept: the document is pending, and is read in the background into
    Markdown that keeps its headings, lists and tables, each passage saying the page it is from. A page
    that is only a scan is read from its picture through the router's ocr-default, billed per page. Poll
    `getKnowledgeDocument` until its state is indexed, or failed with an error saying why: a password, a
    damaged file, or no text at all.
    The passages are keyed by `source`, the file's name unless it is given, so uploading the same source
    again replaces the document; its old passages are still looked up until the new file has been read.
    The file itself is deleted once read. Server-side only.

    Args:
        namespace (str):
        source (str | Unset):
        body (UploadKnowledgeFileBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | IndexedKnowledgeDocument
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            namespace=namespace,
            source=source,
        )
    ).parsed
