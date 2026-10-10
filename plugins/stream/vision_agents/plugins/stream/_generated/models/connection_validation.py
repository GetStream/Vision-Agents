from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connection_validation_status import ConnectionValidationStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectionValidation")


@_attrs_define
class ConnectionValidation:
    """Whether a connection's credential works, found by asking the provider for its tools.

    Attributes:
        connection_id (str):
        status (ConnectionValidationStatus): connected: the credential works and the tools were listed. pending: no
            credentials yet. needs_reauthorization: the provider no longer takes the credential, so only a reconnect helps,
            or, with code connector_credential_rejected, saving credentials again. needs_scopes: the tools were listed, and
            the grant lacks scopes they need; missing_scopes names them, and a consent that asks for them helps. failed: the
            provider could not be reached or listed nothing usable; error says why.
        checked_at (datetime.datetime | Unset): When the tools were listed. Absent until a validate listed them.
        code (str | Unset): What a program branches on when the status is not connected: connector_scope_required with
            needs_scopes; connector_credential_rejected with needs_reauthorization, for a bearer or api_key connection whose
            token or key the provider rejected, or that reads a connector revision marked broken, which only saving
            credentials (PUT .../credentials) fixes. More may be added.
        error (str | Unset): Why the status is not connected, for a person to read.
        missing_scopes (list[str] | None | Unset): With needs_scopes: the scopes the checked tools need that the grant
            lacks, sorted.
        tools_digest (str | Unset): The digest of the tools the connection offers, as GET .../tools shows them. Absent
            until a validate listed them.
    """

    connection_id: str
    status: ConnectionValidationStatus
    checked_at: datetime.datetime | Unset = UNSET
    code: str | Unset = UNSET
    error: str | Unset = UNSET
    missing_scopes: list[str] | None | Unset = UNSET
    tools_digest: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connection_id = self.connection_id

        status = self.status.value

        checked_at: str | Unset = UNSET
        if not isinstance(self.checked_at, Unset):
            checked_at = self.checked_at.isoformat()

        code = self.code

        error = self.error

        missing_scopes: list[str] | None | Unset
        if isinstance(self.missing_scopes, Unset):
            missing_scopes = UNSET
        elif isinstance(self.missing_scopes, list):
            missing_scopes = self.missing_scopes

        else:
            missing_scopes = self.missing_scopes

        tools_digest = self.tools_digest

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connection_id": connection_id,
                "status": status,
            }
        )
        if checked_at is not UNSET:
            field_dict["checked_at"] = checked_at
        if code is not UNSET:
            field_dict["code"] = code
        if error is not UNSET:
            field_dict["error"] = error
        if missing_scopes is not UNSET:
            field_dict["missing_scopes"] = missing_scopes
        if tools_digest is not UNSET:
            field_dict["tools_digest"] = tools_digest

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        connection_id = d.pop("connection_id")

        status = ConnectionValidationStatus(d.pop("status"))

        _checked_at = d.pop("checked_at", UNSET)
        checked_at: datetime.datetime | Unset
        if isinstance(_checked_at, Unset):
            checked_at = UNSET
        else:
            checked_at = datetime.datetime.fromisoformat(_checked_at)

        code = d.pop("code", UNSET)

        error = d.pop("error", UNSET)

        def _parse_missing_scopes(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                missing_scopes_type_0 = cast(list[str], data)

                return missing_scopes_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        missing_scopes = _parse_missing_scopes(d.pop("missing_scopes", UNSET))

        tools_digest = d.pop("tools_digest", UNSET)

        connection_validation = cls(
            connection_id=connection_id,
            status=status,
            checked_at=checked_at,
            code=code,
            error=error,
            missing_scopes=missing_scopes,
            tools_digest=tools_digest,
        )

        connection_validation.additional_properties = d
        return connection_validation

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
