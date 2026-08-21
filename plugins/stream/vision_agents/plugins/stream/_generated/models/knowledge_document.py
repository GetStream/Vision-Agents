from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="KnowledgeDocument")


@_attrs_define
class KnowledgeDocument:
    """
    Attributes:
        source (str): Where the document came from, as a reader would recognise it. Passage ids are keyed by it, so
            posting the same source again replaces what it wrote before.
             Example: pricing.md.
        text (str): The document, whole. It is cut into passages here.
    """

    source: str
    text: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        source = self.source

        text = self.text

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "source": source,
                "text": text,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        source = d.pop("source")

        text = d.pop("text")

        knowledge_document = cls(
            source=source,
            text=text,
        )

        knowledge_document.additional_properties = d
        return knowledge_document

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
