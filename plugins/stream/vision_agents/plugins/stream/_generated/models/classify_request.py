from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.classify_request_questions import ClassifyRequestQuestions
    from ..models.classify_request_tags import ClassifyRequestTags


T = TypeVar("T", bound="ClassifyRequest")


@_attrs_define
class ClassifyRequest:
    """
    Attributes:
        questions (ClassifyRequestQuestions): Keyed by ids of the caller's own choosing, which is how the answers come
            back. An id is not part of what is asked, so a question carries its whole meaning in its instructions.
        state (Any): What the questions are about: a string for plain text, or a JSON object whose parts a question can
            name, such as `message`. Example: I was charged twice this month and nobody has answered my email..
        tags (ClassifyRequestTags | Unset):
        target (str | Unset): A provider/model or a capability shortcut. Empty takes classify-fast. Example: classify-
            fast.
    """

    questions: ClassifyRequestQuestions
    state: Any
    tags: ClassifyRequestTags | Unset = UNSET
    target: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        questions = self.questions.to_dict()

        state = self.state

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        target = self.target

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "questions": questions,
                "state": state,
            }
        )
        if tags is not UNSET:
            field_dict["tags"] = tags
        if target is not UNSET:
            field_dict["target"] = target

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.classify_request_questions import ClassifyRequestQuestions
        from ..models.classify_request_tags import ClassifyRequestTags

        d = dict(src_dict)
        questions = ClassifyRequestQuestions.from_dict(d.pop("questions"))

        state = d.pop("state")

        _tags = d.pop("tags", UNSET)
        tags: ClassifyRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = ClassifyRequestTags.from_dict(_tags)

        target = d.pop("target", UNSET)

        classify_request = cls(
            questions=questions,
            state=state,
            tags=tags,
            target=target,
        )

        classify_request.additional_properties = d
        return classify_request

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
