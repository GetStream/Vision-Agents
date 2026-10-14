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
        assistant_message_id (str):
        duplicate (bool): True when this command already exists and no new inference was started.
        request_id (str):
        state (str): Latest locally recorded response state; an interrupted command is never automatically rerun.
        user_message_id (str):
    """

    assistant_message_id: str
    duplicate: bool
    request_id: str
    state: str
    user_message_id: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        assistant_message_id = self.assistant_message_id

        duplicate = self.duplicate

        request_id = self.request_id

        state = self.state

        user_message_id = self.user_message_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "assistant_message_id": assistant_message_id,
                "duplicate": duplicate,
                "request_id": request_id,
                "state": state,
                "user_message_id": user_message_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        assistant_message_id = d.pop("assistant_message_id")

        duplicate = d.pop("duplicate")

        request_id = d.pop("request_id")

        state = d.pop("state")

        user_message_id = d.pop("user_message_id")

        command_receipt = cls(
            assistant_message_id=assistant_message_id,
            duplicate=duplicate,
            request_id=request_id,
            state=state,
            user_message_id=user_message_id,
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
