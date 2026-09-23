from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="IndexedKnowledgeDocument")


@_attrs_define
class IndexedKnowledgeDocument:
    """
    Attributes:
        id (str):
        namespace (str):
        source (str): What the document was posted as, and what its passages are keyed by. Example: pricing.md.
        passages (int): How many passages it was last cut into.
        created_at (datetime.datetime):
        updated_at (datetime.datetime): When it was last written.
        text (str | Unset): The document as it was last posted. Only reading one document fills it in, and one written
            before its text was kept has none.
    """

    id: str
    namespace: str
    source: str
    passages: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    text: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        namespace = self.namespace

        source = self.source

        passages = self.passages

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        text = self.text

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "namespace": namespace,
                "source": source,
                "passages": passages,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if text is not UNSET:
            field_dict["text"] = text

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        namespace = d.pop("namespace")

        source = d.pop("source")

        passages = d.pop("passages")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        text = d.pop("text", UNSET)

        indexed_knowledge_document = cls(
            id=id,
            namespace=namespace,
            source=source,
            passages=passages,
            created_at=created_at,
            updated_at=updated_at,
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
