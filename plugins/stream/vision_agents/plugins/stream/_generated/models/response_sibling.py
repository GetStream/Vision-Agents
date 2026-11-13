from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ResponseSibling")


@_attrs_define
class ResponseSibling:
    """
    Attributes:
        active (bool): Whether the conversation goes through this sibling now: for a question, through any answer to it.
        created_at (datetime.datetime):
        id (str): The response, which getResponseBranch reads and activateResponse brings back.
        index (int): Its place among the siblings, from 1, oldest first: the i of ‹ i/n ›.
        said (str | Unset): The question the response answers.
    """

    active: bool
    created_at: datetime.datetime
    id: str
    index: int
    said: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        active = self.active

        created_at = self.created_at.isoformat()

        id = self.id

        index = self.index

        said = self.said

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "active": active,
                "created_at": created_at,
                "id": id,
                "index": index,
            }
        )
        if said is not UNSET:
            field_dict["said"] = said

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        active = d.pop("active")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        index = d.pop("index")

        said = d.pop("said", UNSET)

        response_sibling = cls(
            active=active,
            created_at=created_at,
            id=id,
            index=index,
            said=said,
        )

        response_sibling.additional_properties = d
        return response_sibling

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
