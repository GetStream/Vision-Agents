from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ClaimGuestResult")


@_attrs_define
class ClaimGuestResult:
    """
    Attributes:
        guest_id (str):
        user_id (str):
        sessions_moved (int): How many conversations moved onto the account.
    """

    guest_id: str
    user_id: str
    sessions_moved: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        guest_id = self.guest_id

        user_id = self.user_id

        sessions_moved = self.sessions_moved

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "guest_id": guest_id,
                "user_id": user_id,
                "sessions_moved": sessions_moved,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        guest_id = d.pop("guest_id")

        user_id = d.pop("user_id")

        sessions_moved = d.pop("sessions_moved")

        claim_guest_result = cls(
            guest_id=guest_id,
            user_id=user_id,
            sessions_moved=sessions_moved,
        )

        claim_guest_result.additional_properties = d
        return claim_guest_result

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
