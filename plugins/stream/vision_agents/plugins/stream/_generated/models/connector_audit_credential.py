from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorAuditCredential")


@_attrs_define
class ConnectorAuditCredential:
    """The tokens a grant event left, or on grant_revoked the tokens that ended, each named by its fingerprint: the first 4
    bytes of the token's SHA-256, as 8 lowercase hex characters. Two equal fingerprints are the same token, so a refresh
    shows whether the provider rotated the refresh token. No token, and no character of one, is shown.

        Attributes:
            rotated (bool): The refresh token the connection already had was replaced, as a provider that rotates refresh
                tokens does on every refresh.
            access_expires_at (datetime.datetime | Unset): When the access token expires. Absent when the provider did not
                say.
            access_fingerprint (str | Unset): The access token the grant left, by fingerprint. On grant_revoked, the one
                that ended.
            previous_access_fingerprint (str | Unset): The access token before it, by fingerprint. Absent for a first grant.
            previous_refresh_fingerprint (str | Unset): The refresh token before it, by fingerprint. Absent for a first
                grant, or when there was none.
            refresh_expires_at (datetime.datetime | Unset): When the refresh token expires, by the connector's refresh_ttl.
                Absent when it does not say.
            refresh_fingerprint (str | Unset): The refresh token the grant left, by fingerprint. On grant_revoked, the one
                that ended. Absent when there is none.
    """

    rotated: bool
    access_expires_at: datetime.datetime | Unset = UNSET
    access_fingerprint: str | Unset = UNSET
    previous_access_fingerprint: str | Unset = UNSET
    previous_refresh_fingerprint: str | Unset = UNSET
    refresh_expires_at: datetime.datetime | Unset = UNSET
    refresh_fingerprint: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        rotated = self.rotated

        access_expires_at: str | Unset = UNSET
        if not isinstance(self.access_expires_at, Unset):
            access_expires_at = self.access_expires_at.isoformat()

        access_fingerprint = self.access_fingerprint

        previous_access_fingerprint = self.previous_access_fingerprint

        previous_refresh_fingerprint = self.previous_refresh_fingerprint

        refresh_expires_at: str | Unset = UNSET
        if not isinstance(self.refresh_expires_at, Unset):
            refresh_expires_at = self.refresh_expires_at.isoformat()

        refresh_fingerprint = self.refresh_fingerprint

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "rotated": rotated,
            }
        )
        if access_expires_at is not UNSET:
            field_dict["access_expires_at"] = access_expires_at
        if access_fingerprint is not UNSET:
            field_dict["access_fingerprint"] = access_fingerprint
        if previous_access_fingerprint is not UNSET:
            field_dict["previous_access_fingerprint"] = previous_access_fingerprint
        if previous_refresh_fingerprint is not UNSET:
            field_dict["previous_refresh_fingerprint"] = previous_refresh_fingerprint
        if refresh_expires_at is not UNSET:
            field_dict["refresh_expires_at"] = refresh_expires_at
        if refresh_fingerprint is not UNSET:
            field_dict["refresh_fingerprint"] = refresh_fingerprint

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        rotated = d.pop("rotated")

        _access_expires_at = d.pop("access_expires_at", UNSET)
        access_expires_at: datetime.datetime | Unset
        if isinstance(_access_expires_at, Unset):
            access_expires_at = UNSET
        else:
            access_expires_at = datetime.datetime.fromisoformat(_access_expires_at)

        access_fingerprint = d.pop("access_fingerprint", UNSET)

        previous_access_fingerprint = d.pop("previous_access_fingerprint", UNSET)

        previous_refresh_fingerprint = d.pop("previous_refresh_fingerprint", UNSET)

        _refresh_expires_at = d.pop("refresh_expires_at", UNSET)
        refresh_expires_at: datetime.datetime | Unset
        if isinstance(_refresh_expires_at, Unset):
            refresh_expires_at = UNSET
        else:
            refresh_expires_at = datetime.datetime.fromisoformat(_refresh_expires_at)

        refresh_fingerprint = d.pop("refresh_fingerprint", UNSET)

        connector_audit_credential = cls(
            rotated=rotated,
            access_expires_at=access_expires_at,
            access_fingerprint=access_fingerprint,
            previous_access_fingerprint=previous_access_fingerprint,
            previous_refresh_fingerprint=previous_refresh_fingerprint,
            refresh_expires_at=refresh_expires_at,
            refresh_fingerprint=refresh_fingerprint,
        )

        connector_audit_credential.additional_properties = d
        return connector_audit_credential

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
