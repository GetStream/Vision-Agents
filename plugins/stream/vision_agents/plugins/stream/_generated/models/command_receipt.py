from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="CommandReceipt")


@_attrs_define
class CommandReceipt:
    """
    Attributes:
        command_id (str):
        user_message_id (str):
        assistant_message_id (str):
        state (str): Latest locally recorded response state; an interrupted command is never automatically rerun.
        duplicate (bool): True when this command already exists and no new inference was started.
    """

    command_id: str
    user_message_id: str
    assistant_message_id: str
    state: str
    duplicate: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        command_id = self.command_id

        user_message_id = self.user_message_id

        assistant_message_id = self.assistant_message_id

        state = self.state

        duplicate = self.duplicate

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "command_id": command_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "state": state,
                "duplicate": duplicate,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        command_id = d.pop("command_id")

        user_message_id = d.pop("user_message_id")

        assistant_message_id = d.pop("assistant_message_id")

        state = d.pop("state")

        duplicate = d.pop("duplicate")

        command_receipt = cls(
            command_id=command_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            state=state,
            duplicate=duplicate,
        )

        command_receipt.additional_properties = d
        return command_receipt

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
