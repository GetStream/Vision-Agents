from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_memory_filter import SessionMemoryFilter


T = TypeVar("T", bound="SessionMemory")


@_attrs_define
class SessionMemory:
    """Who the session's memories are about. Without a user id nothing is recalled or stored, which is the case for a call
    with nobody identified on it.

        Attributes:
            user_id (str | Unset): Who the memories belong to. Empty means the customer.
            app_id (str | Unset): Separates two deployments sharing one memory account.
            filter_ (SessionMemoryFilter | Unset): The caller's own labels, which narrow recall further. They cannot widen
                it: a filter is applied alongside the user id, never instead of it.
    """

    user_id: str | Unset = UNSET
    app_id: str | Unset = UNSET
    filter_: SessionMemoryFilter | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        app_id = self.app_id

        filter_: dict[str, Any] | Unset = UNSET
        if not isinstance(self.filter_, Unset):
            filter_ = self.filter_.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if user_id is not UNSET:
            field_dict["user_id"] = user_id
        if app_id is not UNSET:
            field_dict["app_id"] = app_id
        if filter_ is not UNSET:
            field_dict["filter"] = filter_

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.session_memory_filter import SessionMemoryFilter

        d = dict(src_dict)
        user_id = d.pop("user_id", UNSET)

        app_id = d.pop("app_id", UNSET)

        _filter_ = d.pop("filter", UNSET)
        filter_: SessionMemoryFilter | Unset
        if isinstance(_filter_, Unset):
            filter_ = UNSET
        else:
            filter_ = SessionMemoryFilter.from_dict(_filter_)

        session_memory = cls(
            user_id=user_id,
            app_id=app_id,
            filter_=filter_,
        )

        session_memory.additional_properties = d
        return session_memory

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
