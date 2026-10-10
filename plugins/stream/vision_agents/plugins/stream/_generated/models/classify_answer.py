from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.classify_question_type import ClassifyQuestionType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.classify_answer_legend import ClassifyAnswerLegend
    from ..models.classify_answer_probabilities import ClassifyAnswerProbabilities


T = TypeVar("T", bound="ClassifyAnswer")


@_attrs_define
class ClassifyAnswer:
    """Which fields carry the answer depends on the type. A noul fills yes alone. A choice fills chosen, probabilities and
    confidence. A score fills level, legend, probabilities and confidence.

        Attributes:
            type_ (ClassifyQuestionType): noul is yes or no, answered as the probability of yes. choice picks one of named
                options. score places the state along ordered levels.
            chosen (str | Unset): The likeliest option of a choice.
            confidence (float | Unset): How peaked the distribution is, not whether acting on it is safe.
            legend (ClassifyAnswerLegend | Unset): A score's levels by index, as decimal strings.
            level (float | Unset): Where a score landed, which may be between two of its levels.
            probabilities (ClassifyAnswerProbabilities | Unset): The distribution the answer came from: options for a
                choice, level indices for a score. They sum to one.
            yes (float | Unset): The probability a noul is true, from 0 to 1.
    """

    type_: ClassifyQuestionType
    chosen: str | Unset = UNSET
    confidence: float | Unset = UNSET
    legend: ClassifyAnswerLegend | Unset = UNSET
    level: float | Unset = UNSET
    probabilities: ClassifyAnswerProbabilities | Unset = UNSET
    yes: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        chosen = self.chosen

        confidence = self.confidence

        legend: dict[str, Any] | Unset = UNSET
        if not isinstance(self.legend, Unset):
            legend = self.legend.to_dict()

        level = self.level

        probabilities: dict[str, Any] | Unset = UNSET
        if not isinstance(self.probabilities, Unset):
            probabilities = self.probabilities.to_dict()

        yes = self.yes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
            }
        )
        if chosen is not UNSET:
            field_dict["chosen"] = chosen
        if confidence is not UNSET:
            field_dict["confidence"] = confidence
        if legend is not UNSET:
            field_dict["legend"] = legend
        if level is not UNSET:
            field_dict["level"] = level
        if probabilities is not UNSET:
            field_dict["probabilities"] = probabilities
        if yes is not UNSET:
            field_dict["yes"] = yes

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.classify_answer_legend import ClassifyAnswerLegend
        from ..models.classify_answer_probabilities import ClassifyAnswerProbabilities

        d = dict(src_dict)
        type_ = ClassifyQuestionType(d.pop("type"))

        chosen = d.pop("chosen", UNSET)

        confidence = d.pop("confidence", UNSET)

        _legend = d.pop("legend", UNSET)
        legend: ClassifyAnswerLegend | Unset
        if isinstance(_legend, Unset):
            legend = UNSET
        else:
            legend = ClassifyAnswerLegend.from_dict(_legend)

        level = d.pop("level", UNSET)

        _probabilities = d.pop("probabilities", UNSET)
        probabilities: ClassifyAnswerProbabilities | Unset
        if isinstance(_probabilities, Unset):
            probabilities = UNSET
        else:
            probabilities = ClassifyAnswerProbabilities.from_dict(_probabilities)

        yes = d.pop("yes", UNSET)

        classify_answer = cls(
            type_=type_,
            chosen=chosen,
            confidence=confidence,
            legend=legend,
            level=level,
            probabilities=probabilities,
            yes=yes,
        )

        classify_answer.additional_properties = d
        return classify_answer

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
