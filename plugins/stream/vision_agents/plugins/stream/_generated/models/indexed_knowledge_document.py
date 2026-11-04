from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.knowledge_document_state import KnowledgeDocumentState
from ..types import UNSET, Unset

T = TypeVar("T", bound="IndexedKnowledgeDocument")


@_attrs_define
class IndexedKnowledgeDocument:
    """
    Attributes:
        bytes_ (int): The size of the file uploaded, or of the text posted. 0 for a document written before sizes were
            kept.
        content_type (str): The type the file was uploaded as, or text/markdown for text posted. Empty for a document
            written before types were kept. Example: application/pdf.
        created_at (datetime.datetime):
        id (str):
        namespace (str):
        pages (int): How many pages the uploaded file has. 0 for text, and for a file without pages, such as a
            spreadsheet.
        passages (int): How many passages it was last cut into. A file uploaded again keeps its old passages, which are
            still looked up, until the new file has been read.
        source (str): What the document was posted as, and what its passages are keyed by. Example: pricing.md.
        state (KnowledgeDocumentState): Where the document has got to. Pending is an uploaded file waiting to be read or
            being read; failed is one that could not be read, and error says why. Text posted is indexed when it is written.
        updated_at (datetime.datetime): When it was last written.
        error (str | Unset): Why the uploaded file could not be read, written for whoever uploaded it. Set while a read
            that failed is tried again, and once it has failed for good.
        text (str | Unset): The document as it was last posted, or as its file was read: Markdown, each page of a file
            opening with an <!-- page N --> comment. Only reading one document fills it in, and one written before its text
            was kept has none.
    """

    bytes_: int
    content_type: str
    created_at: datetime.datetime
    id: str
    namespace: str
    pages: int
    passages: int
    source: str
    state: KnowledgeDocumentState
    updated_at: datetime.datetime
    error: str | Unset = UNSET
    text: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        bytes_ = self.bytes_

        content_type = self.content_type

        created_at = self.created_at.isoformat()

        id = self.id

        namespace = self.namespace

        pages = self.pages

        passages = self.passages

        source = self.source

        state = self.state.value

        updated_at = self.updated_at.isoformat()

        error = self.error

        text = self.text

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "bytes": bytes_,
                "content_type": content_type,
                "created_at": created_at,
                "id": id,
                "namespace": namespace,
                "pages": pages,
                "passages": passages,
                "source": source,
                "state": state,
                "updated_at": updated_at,
            }
        )
        if error is not UNSET:
            field_dict["error"] = error
        if text is not UNSET:
            field_dict["text"] = text

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        bytes_ = d.pop("bytes")

        content_type = d.pop("content_type")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        namespace = d.pop("namespace")

        pages = d.pop("pages")

        passages = d.pop("passages")

        source = d.pop("source")

        state = KnowledgeDocumentState(d.pop("state"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        error = d.pop("error", UNSET)

        text = d.pop("text", UNSET)

        indexed_knowledge_document = cls(
            bytes_=bytes_,
            content_type=content_type,
            created_at=created_at,
            id=id,
            namespace=namespace,
            pages=pages,
            passages=passages,
            source=source,
            state=state,
            updated_at=updated_at,
            error=error,
            text=text,
        )

        indexed_knowledge_document.additional_properties = d
        return indexed_knowledge_document

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
