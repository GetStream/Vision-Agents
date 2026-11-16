from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_change import DataChange


T = TypeVar("T", bound="DataChangePage")


@_attrs_define
class DataChangePage:
    """
    Attributes:
        changes (list[DataChange]):
        cursor (int): What to pass as `after` next time.
        caught_up (bool | Unset): Nothing else has happened yet, which is when a switchover is safe: point your SDKs at
            the new deployment, wait for this to be true once more, and stop.
    """

    changes: list[DataChange]
    cursor: int
    caught_up: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        changes = []
        for changes_item_data in self.changes:
            changes_item = changes_item_data.to_dict()
            changes.append(changes_item)

        cursor = self.cursor

        caught_up = self.caught_up

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "changes": changes,
                "cursor": cursor,
            }
        )
        if caught_up is not UNSET:
            field_dict["caught_up"] = caught_up

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.data_change import DataChange

        d = dict(src_dict)
        changes = []
        _changes = d.pop("changes")
        for changes_item_data in _changes:
            changes_item = DataChange.from_dict(changes_item_data)

            changes.append(changes_item)

        cursor = d.pop("cursor")

        caught_up = d.pop("caught_up", UNSET)

        data_change_page = cls(
            changes=changes,
            cursor=cursor,
            caught_up=caught_up,
        )

        data_change_page.additional_properties = d
        return data_change_page

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
