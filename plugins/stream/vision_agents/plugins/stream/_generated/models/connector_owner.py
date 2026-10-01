from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_owner_type import ConnectorOwnerType
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorOwner")


@_attrs_define
class ConnectorOwner:
    """
    Attributes:
        type_ (ConnectorOwnerType):
        user_id (str | Unset): Required only for a user-owned connection; backend supplied.
    """

    type_: ConnectorOwnerType
    user_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
            }
        )
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        type_ = ConnectorOwnerType(d.pop("type"))

        user_id = d.pop("user_id", UNSET)

        connector_owner = cls(
            type_=type_,
            user_id=user_id,
        )

        connector_owner.additional_properties = d
        return connector_owner

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
