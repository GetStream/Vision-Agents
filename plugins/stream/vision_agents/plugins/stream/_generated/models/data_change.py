from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.data_change_op import DataChangeOp
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_change_key import DataChangeKey
    from ..models.data_change_row import DataChangeRow


T = TypeVar("T", bound="DataChange")


@_attrs_define
class DataChange:
    """One thing that happened to one row of the calling app's data.

    Attributes:
        seq (int): Where this sits in the order changes happened, and the cursor to resume from.
        table (str): Which table the row is in. Example: agent_configs.
        op (DataChangeOp):
        key (DataChangeKey): What identifies the row, which is all a delete has.
        at (datetime.datetime):
        row (DataChangeRow | Unset): The row as it now reads, absent for a delete and never carrying a credential.
    """

    seq: int
    table: str
    op: DataChangeOp
    key: DataChangeKey
    at: datetime.datetime
    row: DataChangeRow | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        seq = self.seq

        table = self.table

        op = self.op.value

        key = self.key.to_dict()

        at = self.at.isoformat()

        row: dict[str, Any] | Unset = UNSET
        if not isinstance(self.row, Unset):
            row = self.row.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "seq": seq,
                "table": table,
                "op": op,
                "key": key,
                "at": at,
            }
        )
        if row is not UNSET:
            field_dict["row"] = row

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.data_change_key import DataChangeKey
        from ..models.data_change_row import DataChangeRow

        d = dict(src_dict)
        seq = d.pop("seq")

        table = d.pop("table")

        op = DataChangeOp(d.pop("op"))

        key = DataChangeKey.from_dict(d.pop("key"))

        at = datetime.datetime.fromisoformat(d.pop("at"))

        _row = d.pop("row", UNSET)
        row: DataChangeRow | Unset
        if isinstance(_row, Unset):
            row = UNSET
        else:
            row = DataChangeRow.from_dict(_row)

        data_change = cls(
            seq=seq,
            table=table,
            op=op,
            key=key,
            at=at,
            row=row,
        )

        data_change.additional_properties = d
        return data_change

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
