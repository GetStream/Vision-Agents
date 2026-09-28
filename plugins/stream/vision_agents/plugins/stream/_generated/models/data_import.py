from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_import_tables import DataImportTables


T = TypeVar("T", bound="DataImport")


@_attrs_define
class DataImport:
    """
    Attributes:
        rows (int): How many rows were written.
        tables (DataImportTables | Unset): How many of them went into each table.
        cursor (int | Unset): The cursor the export named, to ask the other deployment for changes from.
    """

    rows: int
    tables: DataImportTables | Unset = UNSET
    cursor: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        rows = self.rows

        tables: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tables, Unset):
            tables = self.tables.to_dict()

        cursor = self.cursor

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "rows": rows,
            }
        )
        if tables is not UNSET:
            field_dict["tables"] = tables
        if cursor is not UNSET:
            field_dict["cursor"] = cursor

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.data_import_tables import DataImportTables

        d = dict(src_dict)
        rows = d.pop("rows")

        _tables = d.pop("tables", UNSET)
        tables: DataImportTables | Unset
        if isinstance(_tables, Unset):
            tables = UNSET
        else:
            tables = DataImportTables.from_dict(_tables)

        cursor = d.pop("cursor", UNSET)

        data_import = cls(
            rows=rows,
            tables=tables,
            cursor=cursor,
        )

        data_import.additional_properties = d
        return data_import

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
