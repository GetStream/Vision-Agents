from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.guest_user_custom import GuestUserCustom


T = TypeVar("T", bound="GuestUser")


@_attrs_define
class GuestUser:
    """
    Attributes:
        id (str):
        token (str): A Stream user token for this guest, which is what the chat and video SDKs connect with. It carries
            role guest, so an app that has turned guests off refuses it.
        name (str | Unset):
        custom (GuestUserCustom | Unset):
        expires_at (datetime.datetime | Unset): When the token stops working. A guest coming back after it asks for
            another.
    """

    id: str
    token: str
    name: str | Unset = UNSET
    custom: GuestUserCustom | Unset = UNSET
    expires_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        token = self.token

        name = self.name

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        expires_at: str | Unset = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "token": token,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name
        if custom is not UNSET:
            field_dict["custom"] = custom
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.guest_user_custom import GuestUserCustom

        d = dict(src_dict)
        id = d.pop("id")

        token = d.pop("token")

        name = d.pop("name", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: GuestUserCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = GuestUserCustom.from_dict(_custom)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: datetime.datetime | Unset
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = datetime.datetime.fromisoformat(_expires_at)

        guest_user = cls(
            id=id,
            token=token,
            name=name,
            custom=custom,
            expires_at=expires_at,
        )

        guest_user.additional_properties = d
        return guest_user

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
