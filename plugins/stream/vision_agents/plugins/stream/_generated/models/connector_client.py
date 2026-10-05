from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.connector_client_alg import ConnectorClientAlg
from ..models.connector_client_auth_method import ConnectorClientAuthMethod
from ..models.connector_client_source import ConnectorClientSource
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorClient")


@_attrs_define
class ConnectorClient:
    """Where the OAuth client a connection uses may come from, and how the client authenticates at the token endpoint.

    Attributes:
        alg (ConnectorClientAlg | Unset): How a private_key_jwt assertion is signed, and set only for it.
        auth_method (ConnectorClientAuthMethod | Unset): How the OAuth client authenticates at the token endpoint, as
            the IANA OAuth token endpoint authentication methods registry spells it.
        from_ (list[ConnectorClientSource] | None | Unset): Where the OAuth client may come from. Empty when the
            connector needs none.
    """

    alg: ConnectorClientAlg | Unset = UNSET
    auth_method: ConnectorClientAuthMethod | Unset = UNSET
    from_: list[ConnectorClientSource] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        alg: str | Unset = UNSET
        if not isinstance(self.alg, Unset):
            alg = self.alg.value

        auth_method: str | Unset = UNSET
        if not isinstance(self.auth_method, Unset):
            auth_method = self.auth_method.value

        from_: list[str] | None | Unset
        if isinstance(self.from_, Unset):
            from_ = UNSET
        elif isinstance(self.from_, list):
            from_ = []
            for from_type_0_item_data in self.from_:
                from_type_0_item = from_type_0_item_data.value
                from_.append(from_type_0_item)

        else:
            from_ = self.from_

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if alg is not UNSET:
            field_dict["alg"] = alg
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if from_ is not UNSET:
            field_dict["from"] = from_

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

        def _parse_from_(data: object) -> list[ConnectorClientSource] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                from_type_0 = []
                _from_type_0 = data
                for from_type_0_item_data in _from_type_0:
                    from_type_0_item = ConnectorClientSource(from_type_0_item_data)

                    from_type_0.append(from_type_0_item)

                return from_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ConnectorClientSource] | None | Unset, data)

        from_ = _parse_from_(d.pop("from", UNSET))

        connector_client = cls(
            alg=alg,
            auth_method=auth_method,
            from_=from_,
        )

        return connector_client
