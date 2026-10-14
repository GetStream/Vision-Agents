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
        instructions (str):  Example: Is the customer asking for a refund?.
        type_ (ClassifyQuestionType): noul is yes or no, answered as the probability of yes. choice picks one of named
            options. score places the state along ordered levels.
        levels (list[str] | Unset): A score's levels, in order, each describing a concrete situation.
        no (str | Unset): What no means for a noul, where the instructions do not say it.
        options (ClassifyQuestionOptions | Unset): A choice's options, each with a description of what it covers or an
            empty string where the name says it. Include one for "none of these" whenever the options may not cover an
            input.
        yes (str | Unset): What yes means for a noul, where the instructions do not say it.
    """

    instructions: str
    type_: ClassifyQuestionType
    levels: list[str] | Unset = UNSET
    no: str | Unset = UNSET
    options: ClassifyQuestionOptions | Unset = UNSET
    yes: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instructions = self.instructions

        type_ = self.type_.value

        levels: list[str] | Unset = UNSET
        if not isinstance(self.levels, Unset):
            levels = self.levels

        no = self.no

        options: dict[str, Any] | Unset = UNSET
        if not isinstance(self.options, Unset):
            options = self.options.to_dict()

        yes = self.yes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "instructions": instructions,
                "type": type_,
            }
        )
        if levels is not UNSET:
            field_dict["levels"] = levels
        if no is not UNSET:
            field_dict["no"] = no
        if options is not UNSET:
            field_dict["options"] = options
        if yes is not UNSET:
            field_dict["yes"] = yes

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.classify_question_options import ClassifyQuestionOptions

        d = dict(src_dict)
        instructions = d.pop("instructions")

        type_ = ClassifyQuestionType(d.pop("type"))

        levels = cast(list[str], d.pop("levels", UNSET))

        no = d.pop("no", UNSET)

        _options = d.pop("options", UNSET)
        options: ClassifyQuestionOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = ClassifyQuestionOptions.from_dict(_options)

        yes = d.pop("yes", UNSET)

        classify_question = cls(
            instructions=instructions,
            type_=type_,
            levels=levels,
            no=no,
            options=options,
            yes=yes,
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
