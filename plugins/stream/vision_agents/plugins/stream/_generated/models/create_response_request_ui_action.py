from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="CreateResponseRequestUiAction")


@_attrs_define
class CreateResponseRequestUiAction:
    """What the person did in the generated UI of an earlier reply, such as tapping a place or submitting a form, which the
    text says in their words: an object naming the reply (message_id), the UI's format and the action (name), with what
    it carries (payload), of at most 8 KB as JSON. Their message carries it as custom.ai_ui_action, and the model is
    told it after the text.

    """

    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        create_response_request_ui_action = cls()

        create_response_request_ui_action.additional_properties = d
        return create_response_request_ui_action

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
