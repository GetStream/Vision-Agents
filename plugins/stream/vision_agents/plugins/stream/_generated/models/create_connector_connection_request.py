from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_owner import ConnectorOwner


T = TypeVar("T", bound="CreateConnectorConnectionRequest")


@_attrs_define
class CreateConnectorConnectionRequest:
    """
    Attributes:
        connector_id (str):
        owner (ConnectorOwner):
        instance (str | Unset): Provider environment, such as Salesforce production or sandbox.
        label (str | Unset):
    """

    connector_id: str
    owner: ConnectorOwner
    instance: str | Unset = UNSET
    label: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connector_id = self.connector_id

        owner = self.owner.to_dict()

        instance = self.instance

        label = self.label

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connector_id": connector_id,
                "owner": owner,
            }
        )
        if instance is not UNSET:
            field_dict["instance"] = instance
        if label is not UNSET:
            field_dict["label"] = label

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connector_owner import ConnectorOwner

        d = dict(src_dict)
        connector_id = d.pop("connector_id")

        owner = ConnectorOwner.from_dict(d.pop("owner"))

        instance = d.pop("instance", UNSET)

        label = d.pop("label", UNSET)

        create_connector_connection_request = cls(
            connector_id=connector_id,
            owner=owner,
            instance=instance,
            label=label,
        )

        create_connector_connection_request.additional_properties = d
        return create_connector_connection_request

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
