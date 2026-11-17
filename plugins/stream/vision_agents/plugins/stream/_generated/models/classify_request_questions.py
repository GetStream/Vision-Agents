from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.classify_question import ClassifyQuestion


T = TypeVar("T", bound="ClassifyRequestQuestions")


@_attrs_define
class ClassifyRequestQuestions:
    """Keyed by ids of the caller's own choosing, which is how the answers come back. An id is not part of what is asked,
    so a question carries its whole meaning in its instructions.

    """

    additional_properties: dict[str, ClassifyQuestion] = _attrs_field(
        init=False, factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.classify_question import ClassifyQuestion

        d = dict(src_dict)
        classify_request_questions = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():
            additional_property = ClassifyQuestion.from_dict(prop_dict)

            additional_properties[prop_name] = additional_property

        classify_request_questions.additional_properties = additional_properties
        return classify_request_questions

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> ClassifyQuestion:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: ClassifyQuestion) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
