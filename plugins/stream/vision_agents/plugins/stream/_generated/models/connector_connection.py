from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_connection_auth_type import ConnectorConnectionAuthType
from ..models.connector_connection_owner_type import ConnectorConnectionOwnerType
from ..models.connector_connection_status import ConnectorConnectionStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorConnection")


@_attrs_define
class ConnectorConnection:
    """Connection metadata. Credential material is never returned.

    Attributes:
        id (str):
        connector_id (str):
        owner_type (ConnectorConnectionOwnerType):
        endpoint (str):
        auth_type (ConnectorConnectionAuthType):
        status (ConnectorConnectionStatus):
        granted_scopes (list[str]):
        revision (int):
        owner_id (str | Unset):
        instance (str | Unset):
        label (str | Unset):
        account_id (str | Unset):
        expires_at (datetime.datetime | Unset):
        tools_digest (str | Unset):
        tools_checked_at (datetime.datetime | Unset):
        last_error (str | Unset): Sanitized diagnostic; never includes provider response bodies or secrets.
        created_at (datetime.datetime | Unset):
        updated_at (datetime.datetime | Unset):
    """

    id: str
    connector_id: str
    owner_type: ConnectorConnectionOwnerType
    endpoint: str
    auth_type: ConnectorConnectionAuthType
    status: ConnectorConnectionStatus
    granted_scopes: list[str]
    revision: int
    owner_id: str | Unset = UNSET
    instance: str | Unset = UNSET
    label: str | Unset = UNSET
    account_id: str | Unset = UNSET
    expires_at: datetime.datetime | Unset = UNSET
    tools_digest: str | Unset = UNSET
    tools_checked_at: datetime.datetime | Unset = UNSET
    last_error: str | Unset = UNSET
    created_at: datetime.datetime | Unset = UNSET
    updated_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        connector_id = self.connector_id

        owner_type = self.owner_type.value

        endpoint = self.endpoint

        auth_type = self.auth_type.value

        status = self.status.value

        granted_scopes = self.granted_scopes

        revision = self.revision

        owner_id = self.owner_id

        instance = self.instance

        label = self.label

        account_id = self.account_id

        expires_at: str | Unset = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        tools_digest = self.tools_digest

        tools_checked_at: str | Unset = UNSET
        if not isinstance(self.tools_checked_at, Unset):
            tools_checked_at = self.tools_checked_at.isoformat()

        last_error = self.last_error

        created_at: str | Unset = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        updated_at: str | Unset = UNSET
        if not isinstance(self.updated_at, Unset):
            updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "connector_id": connector_id,
                "owner_type": owner_type,
                "endpoint": endpoint,
                "auth_type": auth_type,
                "status": status,
                "granted_scopes": granted_scopes,
                "revision": revision,
            }
        )
        if owner_id is not UNSET:
            field_dict["owner_id"] = owner_id
        if instance is not UNSET:
            field_dict["instance"] = instance
        if label is not UNSET:
            field_dict["label"] = label
        if account_id is not UNSET:
            field_dict["account_id"] = account_id
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at
        if tools_digest is not UNSET:
            field_dict["tools_digest"] = tools_digest
        if tools_checked_at is not UNSET:
            field_dict["tools_checked_at"] = tools_checked_at
        if last_error is not UNSET:
            field_dict["last_error"] = last_error
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        connector_id = d.pop("connector_id")

        owner_type = ConnectorConnectionOwnerType(d.pop("owner_type"))

        endpoint = d.pop("endpoint")

        auth_type = ConnectorConnectionAuthType(d.pop("auth_type"))

        status = ConnectorConnectionStatus(d.pop("status"))

        granted_scopes = cast(list[str], d.pop("granted_scopes"))

        revision = d.pop("revision")

        owner_id = d.pop("owner_id", UNSET)

        instance = d.pop("instance", UNSET)

        label = d.pop("label", UNSET)

        account_id = d.pop("account_id", UNSET)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: datetime.datetime | Unset
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = datetime.datetime.fromisoformat(_expires_at)

        tools_digest = d.pop("tools_digest", UNSET)

        _tools_checked_at = d.pop("tools_checked_at", UNSET)
        tools_checked_at: datetime.datetime | Unset
        if isinstance(_tools_checked_at, Unset):
            tools_checked_at = UNSET
        else:
            tools_checked_at = datetime.datetime.fromisoformat(_tools_checked_at)

        last_error = d.pop("last_error", UNSET)

        _created_at = d.pop("created_at", UNSET)
        created_at: datetime.datetime | Unset
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = datetime.datetime.fromisoformat(_created_at)

        _updated_at = d.pop("updated_at", UNSET)
        updated_at: datetime.datetime | Unset
        if isinstance(_updated_at, Unset):
            updated_at = UNSET
        else:
            updated_at = datetime.datetime.fromisoformat(_updated_at)

        connector_connection = cls(
            id=id,
            connector_id=connector_id,
            owner_type=owner_type,
            endpoint=endpoint,
            auth_type=auth_type,
            status=status,
            granted_scopes=granted_scopes,
            revision=revision,
            owner_id=owner_id,
            instance=instance,
            label=label,
            account_id=account_id,
            expires_at=expires_at,
            tools_digest=tools_digest,
            tools_checked_at=tools_checked_at,
            last_error=last_error,
            created_at=created_at,
            updated_at=updated_at,
        )

        connector_connection.additional_properties = d
        return connector_connection

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
