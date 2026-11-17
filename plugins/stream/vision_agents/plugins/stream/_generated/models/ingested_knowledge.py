from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="IngestedKnowledge")


@_attrs_define
class IngestedKnowledge:
    """
    Attributes:
        documents (int): How many documents were read.
        namespace (str):
        passages (int): How many passages they were cut into and written as.
    """

    documents: int
    namespace: str
    passages: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        documents = self.documents

        namespace = self.namespace

        passages = self.passages

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "documents": documents,
                "namespace": namespace,
                "passages": passages,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        documents = d.pop("documents")

        namespace = d.pop("namespace")

        passages = d.pop("passages")

        ingested_knowledge = cls(
            documents=documents,
            namespace=namespace,
            passages=passages,
        )

        ingested_knowledge.additional_properties = d
        return ingested_knowledge

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
