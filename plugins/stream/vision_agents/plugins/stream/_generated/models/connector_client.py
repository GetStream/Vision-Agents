from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.connector_client_alg import ConnectorClientAlg
from ..models.connector_client_auth_method import ConnectorClientAuthMethod
from ..models.connector_client_owner import ConnectorClientOwner
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorClient")


@_attrs_define
class ConnectorClient:
    """Who may own the OAuth client a connection uses, and how the client authenticates at the token endpoint.

    Attributes:
        alg (ConnectorClientAlg | Unset): How a private_key_jwt assertion is signed, and set only for it.
        auth_method (ConnectorClientAuthMethod | Unset): How the OAuth client authenticates at the token endpoint, as
            the IANA OAuth token endpoint authentication methods registry spells it.
        policy (list[ConnectorClientOwner] | None | Unset): Who may own the OAuth client. Empty when the connector needs
            none.
    """

    alg: ConnectorClientAlg | Unset = UNSET
    auth_method: ConnectorClientAuthMethod | Unset = UNSET
    policy: list[ConnectorClientOwner] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        alg: str | Unset = UNSET
        if not isinstance(self.alg, Unset):
            alg = self.alg.value

        auth_method: str | Unset = UNSET
        if not isinstance(self.auth_method, Unset):
            auth_method = self.auth_method.value

        policy: list[str] | None | Unset
        if isinstance(self.policy, Unset):
            policy = UNSET
        elif isinstance(self.policy, list):
            policy = []
            for policy_type_0_item_data in self.policy:
                policy_type_0_item = policy_type_0_item_data.value
                policy.append(policy_type_0_item)

        else:
            policy = self.policy

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if alg is not UNSET:
            field_dict["alg"] = alg
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if policy is not UNSET:
            field_dict["policy"] = policy

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _alg = d.pop("alg", UNSET)
        alg: ConnectorClientAlg | Unset
        if isinstance(_alg, Unset):
            alg = UNSET
        else:
            alg = ConnectorClientAlg(_alg)

        _auth_method = d.pop("auth_method", UNSET)
        auth_method: ConnectorClientAuthMethod | Unset
        if isinstance(_auth_method, Unset):
            auth_method = UNSET
        else:
            auth_method = ConnectorClientAuthMethod(_auth_method)

        def _parse_policy(data: object) -> list[ConnectorClientOwner] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                policy_type_0 = []
                _policy_type_0 = data
                for policy_type_0_item_data in _policy_type_0:
                    policy_type_0_item = ConnectorClientOwner(policy_type_0_item_data)

                    policy_type_0.append(policy_type_0_item)

                return policy_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ConnectorClientOwner] | None | Unset, data)

        policy = _parse_policy(d.pop("policy", UNSET))

        connector_client = cls(
            alg=alg,
            auth_method=auth_method,
            policy=policy,
        )

        return connector_client
