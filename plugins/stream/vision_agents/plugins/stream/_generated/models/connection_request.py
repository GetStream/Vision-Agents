from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connection_owner import ConnectionOwner
    from ..models.connection_request_inputs import ConnectionRequestInputs


T = TypeVar("T", bound="ConnectionRequest")


@_attrs_define
class ConnectionRequest:
    """A connection to create, pending until an account is connected. An unknown field is refused rather than ignored.

    Attributes:
        connector_id (str): A built-in, such as slack, or one of the app's own.
        owner (ConnectionOwner): Whose a connection is: the app's, which any of its agents may be bound to, or one
            user's.
        auth_scheme (str | Unset): One of the connector's schemes. Omitted is its only one; a connector with several
            needs it named.
        inputs (ConnectionRequestInputs | Unset): Values for the connector's inputs, such as a region. One without a
            default is required, and each must match the connector's enum or pattern.
        label (str | Unset): A name to tell connections apart by.
    """

    connector_id: str
    owner: ConnectionOwner
    auth_scheme: str | Unset = UNSET
    inputs: ConnectionRequestInputs | Unset = UNSET
    label: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        connector_id = self.connector_id

        owner = self.owner.to_dict()

        auth_scheme = self.auth_scheme

        inputs: dict[str, Any] | Unset = UNSET
        if not isinstance(self.inputs, Unset):
            inputs = self.inputs.to_dict()

        label = self.label

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "connector_id": connector_id,
                "owner": owner,
            }
        )
        if auth_scheme is not UNSET:
            field_dict["auth_scheme"] = auth_scheme
        if inputs is not UNSET:
            field_dict["inputs"] = inputs
        if label is not UNSET:
            field_dict["label"] = label

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connection_owner import ConnectionOwner
        from ..models.connection_request_inputs import (
            ConnectionRequestInputs,
        )

        d = dict(src_dict)
        connector_id = d.pop("connector_id")

        owner = ConnectionOwner.from_dict(d.pop("owner"))

        auth_scheme = d.pop("auth_scheme", UNSET)

        _inputs = d.pop("inputs", UNSET)
        inputs: ConnectionRequestInputs | Unset
        if isinstance(_inputs, Unset):
            inputs = UNSET
        else:
            inputs = ConnectionRequestInputs.from_dict(_inputs)

        label = d.pop("label", UNSET)

        connection_request = cls(
            connector_id=connector_id,
            owner=owner,
            auth_scheme=auth_scheme,
            inputs=inputs,
            label=label,
        )

        return connection_request
