from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.knowledge_document import KnowledgeDocument


T = TypeVar("T", bound="IngestKnowledgeRequest")


@_attrs_define
class IngestKnowledgeRequest:
    """
    Attributes:
        namespace (str): The knowledge base to write into, which is what a config's knowledge_namespace names. Knowledge
            is never shared, so there is no default.
             Example: docs.
        documents (list[KnowledgeDocument]):
        chunk_size (int | Unset): Characters per passage. Zero is the default, which is small enough that several
            passages fit in front of a model and large enough that one still answers the question on its own.
    """

    namespace: str
    documents: list[KnowledgeDocument]
    chunk_size: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        namespace = self.namespace

        documents = []
        for documents_item_data in self.documents:
            documents_item = documents_item_data.to_dict()
            documents.append(documents_item)

        chunk_size = self.chunk_size

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "namespace": namespace,
                "documents": documents,
            }
        )
        if chunk_size is not UNSET:
            field_dict["chunk_size"] = chunk_size

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.knowledge_document import KnowledgeDocument

        d = dict(src_dict)
        namespace = d.pop("namespace")

        documents = []
        _documents = d.pop("documents")
        for documents_item_data in _documents:
            documents_item = KnowledgeDocument.from_dict(documents_item_data)

            documents.append(documents_item)

        chunk_size = d.pop("chunk_size", UNSET)

        ingest_knowledge_request = cls(
            namespace=namespace,
            documents=documents,
            chunk_size=chunk_size,
        )

        ingest_knowledge_request.additional_properties = d
        return ingest_knowledge_request

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
