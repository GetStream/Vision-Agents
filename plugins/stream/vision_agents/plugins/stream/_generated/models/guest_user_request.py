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
        id (str | Unset): A guest coming back, to be given a fresh token as the same guest. Only honoured for a caller
            that proves it is that guest: the app's backend, the guest's own still-valid token, or the refresh_secret they
            were last handed. Any other caller is minted a new guest, as if no id had been sent, because a guest id is no
            secret: it is the author of their messages. Omitted mints a new guest.
        name (str | Unset): What to call a new guest, for a transcript a person reads later. A guest coming back keeps
            their name unless the app's backend renames them.
        refresh_secret (str | Unset): The refresh_secret this guest was last handed, sent with their id once their token
            has expired. Spent by being used. One that is wrong or already spent mints a new guest instead.
    """

    custom: GuestUserRequestCustom | Unset = UNSET
    id: str | Unset = UNSET
    name: str | Unset = UNSET
    refresh_secret: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        id = self.id

        name = self.name

        refresh_secret = self.refresh_secret

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if custom is not UNSET:
            field_dict["custom"] = custom
        if id is not UNSET:
            field_dict["id"] = id
        if name is not UNSET:
            field_dict["name"] = name
        if refresh_secret is not UNSET:
            field_dict["refresh_secret"] = refresh_secret

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.guest_user_request_custom import GuestUserRequestCustom

        d = dict(src_dict)
        _custom = d.pop("custom", UNSET)
        custom: GuestUserRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = GuestUserRequestCustom.from_dict(_custom)

        id = d.pop("id", UNSET)

        name = d.pop("name", UNSET)

        refresh_secret = d.pop("refresh_secret", UNSET)

        guest_user_request = cls(
            custom=custom,
            id=id,
            name=name,
            refresh_secret=refresh_secret,
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
