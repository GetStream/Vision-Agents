from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.history_role import HistoryRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="HistoryMessage")


@_attrs_define
class HistoryMessage:
    """
    Attributes:
        role (HistoryRole): user is what a person said, assistant what the agent answered. These are the only turns a
            resumed conversation hands the model; instructions say anything a system message would.
        text (str):
        created_at (datetime.datetime | Unset): When it was said. The model is shown it beside a person's message, so it
            can tell an hour ago from just now.
        name (str | Unset): Who said it, when several people share the thread. The model is shown it as a label, never
            as who is asking now.
    """

    role: HistoryRole
    text: str
    created_at: datetime.datetime | Unset = UNSET
    name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        role = self.role.value

        text = self.text

        created_at: str | Unset = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "role": role,
                "text": text,
            }
        )
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        role = HistoryRole(d.pop("role"))

        text = d.pop("text")

        _created_at = d.pop("created_at", UNSET)
        created_at: datetime.datetime | Unset
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = datetime.datetime.fromisoformat(_created_at)

        name = d.pop("name", UNSET)

        history_message = cls(
            role=role,
            text=text,
            created_at=created_at,
            name=name,
        )

        history_message.additional_properties = d
        return history_message

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
