from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.branch_message_attachments_item import BranchMessageAttachmentsItem
    from ..models.branch_message_custom import BranchMessageCustom


T = TypeVar("T", bound="BranchMessage")


@_attrs_define
class BranchMessage:
    """
    Attributes:
        attachments (list[BranchMessageAttachmentsItem]):
        created_at (datetime.datetime):
        custom (BranchMessageCustom): The message's custom data, as Stream Chat held it, with support_branch saying
            where it stands among its alternatives now.
        id (str): The id the message had in Stream Chat.
        text (str):
        user_id (str): Who wrote it: the person, or the agent's Stream user.
        role (str | Unset): user or assistant, for a message the router wrote. Absent for one somebody else wrote in the
            channel.
    """

    attachments: list[BranchMessageAttachmentsItem]
    created_at: datetime.datetime
    custom: BranchMessageCustom
    id: str
    text: str
    user_id: str
    role: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        attachments = []
        for attachments_item_data in self.attachments:
            attachments_item = attachments_item_data.to_dict()
            attachments.append(attachments_item)

        created_at = self.created_at.isoformat()

        custom = self.custom.to_dict()

        id = self.id

        text = self.text

        user_id = self.user_id

        role = self.role

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "attachments": attachments,
                "created_at": created_at,
                "custom": custom,
                "id": id,
                "text": text,
                "user_id": user_id,
            }
        )
        if role is not UNSET:
            field_dict["role"] = role

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.branch_message_attachments_item import (
            BranchMessageAttachmentsItem,
        )
        from ..models.branch_message_custom import BranchMessageCustom

        d = dict(src_dict)
        attachments = []
        _attachments = d.pop("attachments")
        for attachments_item_data in _attachments:
            attachments_item = BranchMessageAttachmentsItem.from_dict(
                attachments_item_data
            )

            attachments.append(attachments_item)

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        custom = BranchMessageCustom.from_dict(d.pop("custom"))

        id = d.pop("id")

        text = d.pop("text")

        user_id = d.pop("user_id")

        role = d.pop("role", UNSET)

        branch_message = cls(
            attachments=attachments,
            created_at=created_at,
            custom=custom,
            id=id,
            text=text,
            user_id=user_id,
            role=role,
        )

        branch_message.additional_properties = d
        return branch_message

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
