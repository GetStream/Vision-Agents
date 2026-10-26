from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.credit_entry_kind import CreditEntryKind
from ..types import UNSET, Unset

T = TypeVar("T", bound="CreditEntry")


@_attrs_define
class CreditEntry:
    """
    Attributes:
        amount_micros (int): Signed: what the entry added to the credit, in millionths of a dollar.
        created_at (datetime.datetime):
        id (int):
        kind (CreditEntryKind):
        reference (str | Unset): The payment's own reference, such as a top-up's payment id. Absent on entries Stream's
            staff wrote.
    """

    amount_micros: int
    created_at: datetime.datetime
    id: int
    kind: CreditEntryKind
    reference: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        amount_micros = self.amount_micros

        created_at = self.created_at.isoformat()

        id = self.id

        kind = self.kind.value

        reference = self.reference

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "amount_micros": amount_micros,
                "created_at": created_at,
                "id": id,
                "kind": kind,
            }
        )
        if reference is not UNSET:
            field_dict["reference"] = reference

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        amount_micros = d.pop("amount_micros")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        kind = CreditEntryKind(d.pop("kind"))

        reference = d.pop("reference", UNSET)

        credit_entry = cls(
            amount_micros=amount_micros,
            created_at=created_at,
            id=id,
            kind=kind,
            reference=reference,
        )

        credit_entry.additional_properties = d
        return credit_entry

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
