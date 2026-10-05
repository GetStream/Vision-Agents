from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_connection_status import ConnectorConnectionStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_inputs import ConnectionInputs
    from ..models.connection_metadata import ConnectionMetadata
    from ..models.connector_connection_owner import ConnectorConnectionOwner


T = TypeVar("T", bound="Connection")


@_attrs_define
class Connection:
    """One account at one connector, owned by the app or by one of its users. Credentials are never shown.

    Attributes:
        auth_scheme (str): How the connection authenticates, one of its connector's schemes.
        connector_id (str):
        created_at (datetime.datetime):
        definition_revision (int): The connector's revision when the connection was made, which it keeps reading until
            it is reconnected.
        granted_scopes (list[str] | None):
        id (str):
        inputs (ConnectionInputs): What the connection was created with, the connector's defaults filled in.
        metadata (ConnectionMetadata): What the provider said about the account when it was connected, such as a
            workspace id. Empty until then.
        owner (ConnectorConnectionOwner): Whose a connection is: the app's, which any of its agents may be bound to, or
            one user's.
        revision (int): Advances with every new credential, starting at 1.
        status (ConnectorConnectionStatus): pending until an account is connected, then connected, needs_reauthorization
            once the provider stops accepting its credential, and disconnected when it is deleted.
        updated_at (datetime.datetime):
        account_id (str | Unset): The provider account, known once it is connected.
        expires_at (datetime.datetime | Unset): When the current credential expires. Absent when there is none or it
            does not.
        label (str | Unset):
    """

    auth_scheme: str
    connector_id: str
    created_at: datetime.datetime
    definition_revision: int
    granted_scopes: list[str] | None
    id: str
    inputs: ConnectionInputs
    metadata: ConnectionMetadata
    owner: ConnectorConnectionOwner
    revision: int
    status: ConnectorConnectionStatus
    updated_at: datetime.datetime
    account_id: str | Unset = UNSET
    expires_at: datetime.datetime | Unset = UNSET
    label: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        auth_scheme = self.auth_scheme

        connector_id = self.connector_id

        created_at = self.created_at.isoformat()

        definition_revision = self.definition_revision

        granted_scopes: list[str] | None
        if isinstance(self.granted_scopes, list):
            granted_scopes = self.granted_scopes

        else:
            granted_scopes = self.granted_scopes

        id = self.id

        inputs = self.inputs.to_dict()

        metadata = self.metadata.to_dict()

        owner = self.owner.to_dict()

        revision = self.revision

        status = self.status.value

        updated_at = self.updated_at.isoformat()

        account_id = self.account_id

        expires_at: str | Unset = UNSET
        if not isinstance(self.expires_at, Unset):
            expires_at = self.expires_at.isoformat()

        label = self.label

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "auth_scheme": auth_scheme,
                "connector_id": connector_id,
                "created_at": created_at,
                "definition_revision": definition_revision,
                "granted_scopes": granted_scopes,
                "id": id,
                "inputs": inputs,
                "metadata": metadata,
                "owner": owner,
                "revision": revision,
                "status": status,
                "updated_at": updated_at,
            }
        )
        if account_id is not UNSET:
            field_dict["account_id"] = account_id
        if expires_at is not UNSET:
            field_dict["expires_at"] = expires_at
        if label is not UNSET:
            field_dict["label"] = label

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connection_inputs import ConnectionInputs
        from ..models.connection_metadata import ConnectionMetadata
        from ..models.connector_connection_owner import (
            ConnectorConnectionOwner,
        )

        d = dict(src_dict)
        auth_scheme = d.pop("auth_scheme")

        connector_id = d.pop("connector_id")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        definition_revision = d.pop("definition_revision")

        def _parse_granted_scopes(data: object) -> list[str] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                granted_scopes_type_0 = cast(list[str], data)

                return granted_scopes_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None, data)

        granted_scopes = _parse_granted_scopes(d.pop("granted_scopes"))

        id = d.pop("id")

        inputs = ConnectionInputs.from_dict(d.pop("inputs"))

        metadata = ConnectionMetadata.from_dict(d.pop("metadata"))

        owner = ConnectorConnectionOwner.from_dict(d.pop("owner"))

        revision = d.pop("revision")

        status = ConnectorConnectionStatus(d.pop("status"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        account_id = d.pop("account_id", UNSET)

        _expires_at = d.pop("expires_at", UNSET)
        expires_at: datetime.datetime | Unset
        if isinstance(_expires_at, Unset):
            expires_at = UNSET
        else:
            expires_at = datetime.datetime.fromisoformat(_expires_at)

        label = d.pop("label", UNSET)

        connection = cls(
            auth_scheme=auth_scheme,
            connector_id=connector_id,
            created_at=created_at,
            definition_revision=definition_revision,
            granted_scopes=granted_scopes,
            id=id,
            inputs=inputs,
            metadata=metadata,
            owner=owner,
            revision=revision,
            status=status,
            updated_at=updated_at,
            account_id=account_id,
            expires_at=expires_at,
            label=label,
        )

        connection.additional_properties = d
        return connection

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
