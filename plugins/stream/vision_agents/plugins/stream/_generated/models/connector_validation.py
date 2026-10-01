from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_validation_status import ConnectorValidationStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorValidation")


@_attrs_define
class ConnectorValidation:
    """
    Attributes:
        connection_id (str):
        status (ConnectorValidationStatus):
        tools_digest (str):
        checked_at (datetime.datetime | Unset):
        error (str | Unset):
    """

    connection_id: str
    status: ConnectorValidationStatus
    tools_digest: str
    checked_at: datetime.datetime | Unset = UNSET
    error: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        status = self.status.value

        tools_digest = self.tools_digest

        checked_at: str | Unset = UNSET
        if not isinstance(self.checked_at, Unset):
            checked_at = self.checked_at.isoformat()

        error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "status": status,
                "tools_digest": tools_digest,
            }
        )
        if checked_at is not UNSET:
            field_dict["checked_at"] = checked_at
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        status = ConnectorValidationStatus(d.pop("status"))

        tools_digest = d.pop("tools_digest")

        _checked_at = d.pop("checked_at", UNSET)
        checked_at: datetime.datetime | Unset
        if isinstance(_checked_at, Unset):
            checked_at = UNSET
        else:
            checked_at = datetime.datetime.fromisoformat(_checked_at)

        error = d.pop("error", UNSET)

        connector_validation = cls(
            connection_id=connection_id,
            status=status,
            tools_digest=tools_digest,
            checked_at=checked_at,
            error=error,
        )

        connector_validation.additional_properties = d
        return connector_validation

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
