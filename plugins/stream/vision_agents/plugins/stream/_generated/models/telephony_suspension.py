from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="TelephonySuspension")


@_attrs_define
class TelephonySuspension:
    """
    Attributes:
        customer_id (str): The app that is suspended.
        reason (str): Why, for Stream staff. The app is never shown it.
        suspended_at (datetime.datetime): When it was suspended, or last changed.
        suspended_by (str): Who suspended it.
    """

    customer_id: str
    reason: str
    suspended_at: datetime.datetime
    suspended_by: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        customer_id = self.customer_id

        reason = self.reason

        suspended_at = self.suspended_at.isoformat()

        suspended_by = self.suspended_by

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "customer_id": customer_id,
                "reason": reason,
                "suspended_at": suspended_at,
                "suspended_by": suspended_by,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        customer_id = d.pop("customer_id")

        reason = d.pop("reason")

        suspended_at = datetime.datetime.fromisoformat(d.pop("suspended_at"))

        suspended_by = d.pop("suspended_by")

        telephony_suspension = cls(
            customer_id=customer_id,
            reason=reason,
            suspended_at=suspended_at,
            suspended_by=suspended_by,
        )

        telephony_suspension.additional_properties = d
        return telephony_suspension

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
