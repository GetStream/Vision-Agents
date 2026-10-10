from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connection_validation_status import ConnectionValidationStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectionLastValidation")


@_attrs_define
class ConnectionLastValidation:
    """What a connection's last validate found, kept so it is still shown after the validate's answer is gone. A validate
    whose provider refused a bearer or api_key credential with any 4xx but 429 also moves the connection to
    needs_reauthorization.

        Attributes:
            checked_at (datetime.datetime): When the validate ran.
            status (ConnectionValidationStatus): connected: the credential works and the tools were listed. pending: no
                credentials yet. needs_reauthorization: the provider no longer takes the credential, so only a reconnect helps,
                or, with code connector_credential_rejected, saving credentials again. needs_scopes: the tools were listed, and
                the grant lacks scopes they need; missing_scopes names them, and a consent that asks for them helps. failed: the
                provider could not be reached or listed nothing usable; error says why.
            code (str | Unset): The validate's code (connector_credential_rejected, connector_scope_required) when it had
                one. Otherwise, when the provider's last answer was an HTTP error, its status, such as 400 or 503. Absent when
                neither applies.
            error (str | Unset): Why the status is not connected, for a person to read: the validate's error with every
                value the credential is sent as cut out, and cut at 1 KiB. A provider's own error text in it can still hold
                anything else the provider wrote.
    """

    checked_at: datetime.datetime
    status: ConnectionValidationStatus
    code: str | Unset = UNSET
    error: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        checked_at = self.checked_at.isoformat()

        status = self.status.value

        code = self.code

        error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "checked_at": checked_at,
                "status": status,
            }
        )
        if code is not UNSET:
            field_dict["code"] = code
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        checked_at = datetime.datetime.fromisoformat(d.pop("checked_at"))

        status = ConnectionValidationStatus(d.pop("status"))

        code = d.pop("code", UNSET)

        error = d.pop("error", UNSET)

        connection_last_validation = cls(
            checked_at=checked_at,
            status=status,
            code=code,
            error=error,
        )

        connection_last_validation.additional_properties = d
        return connection_last_validation

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
