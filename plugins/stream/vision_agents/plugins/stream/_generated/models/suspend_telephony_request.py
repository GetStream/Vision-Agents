from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="SuspendTelephonyRequest")


@_attrs_define
class SuspendTelephonyRequest:
    """
    Attributes:
        reason (str): Why the app is suspended. The app is never shown it.
        suspended_by (str): Who is suspending it.
    """

    reason: str
    suspended_by: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reason = self.reason

        suspended_by = self.suspended_by

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "reason": reason,
                "suspended_by": suspended_by,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        reason = d.pop("reason")

        suspended_by = d.pop("suspended_by")

        suspend_telephony_request = cls(
            reason=reason,
            suspended_by=suspended_by,
        )

        suspend_telephony_request.additional_properties = d
        return suspend_telephony_request

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
