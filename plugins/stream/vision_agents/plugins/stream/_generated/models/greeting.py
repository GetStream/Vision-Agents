from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.greeting_mode import GreetingMode
from ..types import UNSET, Unset

T = TypeVar("T", bound="Greeting")


@_attrs_define
class Greeting:
    """What the agent says as it joins, before anyone speaks.

    Attributes:
        text (str): What the agent says on joining. Empty means the agent waits to be spoken to.
        mode (GreetingMode | Unset): exact says the text word for word. variation has the model say its own variation of
            it on every call, so callers do not hear the same opening each time. A speech-to-speech model cannot say exact
            words, so it always says its own rendering of the text.
    """

    text: str
    mode: GreetingMode | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
            }
        )
        if mode is not UNSET:
            field_dict["mode"] = mode

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        text = d.pop("text")

        _mode = d.pop("mode", UNSET)
        mode: GreetingMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = GreetingMode(_mode)

        greeting = cls(
            text=text,
            mode=mode,
        )

        greeting.additional_properties = d
        return greeting

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
