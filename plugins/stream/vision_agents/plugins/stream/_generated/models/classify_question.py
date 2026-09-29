from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.classify_question_type import ClassifyQuestionType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.classify_question_options import ClassifyQuestionOptions


T = TypeVar("T", bound="ClassifyQuestion")


@_attrs_define
class ClassifyQuestion:
    """
    Attributes:
        type_ (ClassifyQuestionType): noul is yes or no, answered as the probability of yes. choice picks one of named
            options. score places the state along ordered levels.
        instructions (str):  Example: Is the customer asking for a refund?.
        options (ClassifyQuestionOptions | Unset): A choice's options, each with a description of what it covers or an
            empty string where the name says it. Include one for "none of these" whenever the options may not cover an
            input.
        levels (list[str] | Unset): A score's levels, in order, each describing a concrete situation.
        yes (str | Unset): What yes means for a noul, where the instructions do not say it.
        no (str | Unset): What no means for a noul, where the instructions do not say it.
    """

    type_: ClassifyQuestionType
    instructions: str
    options: ClassifyQuestionOptions | Unset = UNSET
    levels: list[str] | Unset = UNSET
    yes: str | Unset = UNSET
    no: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        instructions = self.instructions

        options: dict[str, Any] | Unset = UNSET
        if not isinstance(self.options, Unset):
            options = self.options.to_dict()

        levels: list[str] | Unset = UNSET
        if not isinstance(self.levels, Unset):
            levels = self.levels

        yes = self.yes

        no = self.no

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
                "instructions": instructions,
            }
        )
        if options is not UNSET:
            field_dict["options"] = options
        if levels is not UNSET:
            field_dict["levels"] = levels
        if yes is not UNSET:
            field_dict["yes"] = yes
        if no is not UNSET:
            field_dict["no"] = no

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.classify_question_options import (
            ClassifyQuestionOptions,
        )

        d = dict(src_dict)
        type_ = ClassifyQuestionType(d.pop("type"))

        instructions = d.pop("instructions")

        _options = d.pop("options", UNSET)
        options: ClassifyQuestionOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = ClassifyQuestionOptions.from_dict(_options)

        levels = cast(list[str], d.pop("levels", UNSET))

        yes = d.pop("yes", UNSET)

        no = d.pop("no", UNSET)

        classify_question = cls(
            type_=type_,
            instructions=instructions,
            options=options,
            levels=levels,
            yes=yes,
            no=no,
        )

        classify_question.additional_properties = d
        return classify_question

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
