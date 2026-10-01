from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.classify_result_answers import ClassifyResultAnswers
    from ..models.classify_usage import ClassifyUsage


T = TypeVar("T", bound="ClassifyResult")


@_attrs_define
class ClassifyResult:
    """
    Attributes:
        provider (str):
        model (str): The version that answered, which is worth recording when the target was an alias.
        answers (ClassifyResultAnswers):
        usage (ClassifyUsage): What the request read and wrote. The state's tokens are counted once however many
            questions shared them.
    """

    provider: str
    model: str
    answers: ClassifyResultAnswers
    usage: ClassifyUsage
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        model = self.model

        answers = self.answers.to_dict()

        usage = self.usage.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "model": model,
                "answers": answers,
                "usage": usage,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.classify_result_answers import (
            ClassifyResultAnswers,
        )
        from ..models.classify_usage import ClassifyUsage

        d = dict(src_dict)
        provider = d.pop("provider")

        model = d.pop("model")

        answers = ClassifyResultAnswers.from_dict(d.pop("answers"))

        usage = ClassifyUsage.from_dict(d.pop("usage"))

        classify_result = cls(
            provider=provider,
            model=model,
            answers=answers,
            usage=usage,
        )

        classify_result.additional_properties = d
        return classify_result

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
