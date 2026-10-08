from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.guest_user_request_custom import GuestUserRequestCustom


T = TypeVar("T", bound="GuestUserRequest")


@_attrs_define
class GuestUserRequest:
    """
    Attributes:
        custom (GuestUserRequestCustom | Unset):
        id (str | Unset): A guest id to reuse, for somebody coming back. Omitted mints a new one. Asking for an id that
            is already a guest of this customer returns that guest with a fresh token rather than failing, because coming
            back is the same person.
        name (str | Unset): What to call them, for a transcript a person reads later.
    """

    custom: GuestUserRequestCustom | Unset = UNSET
    id: str | Unset = UNSET
    name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        id = self.id

        name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if custom is not UNSET:
            field_dict["custom"] = custom
        if id is not UNSET:
            field_dict["id"] = id
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.guest_user_request_custom import (
            GuestUserRequestCustom,
        )

        d = dict(src_dict)
        _custom = d.pop("custom", UNSET)
        custom: GuestUserRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = GuestUserRequestCustom.from_dict(_custom)

        id = d.pop("id", UNSET)

        name = d.pop("name", UNSET)

        guest_user_request = cls(
            custom=custom,
            id=id,
            name=name,
        )

        guest_user_request.additional_properties = d
        return guest_user_request

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
