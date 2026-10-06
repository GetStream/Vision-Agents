from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.stream_key_input import StreamKeyInput


T = TypeVar("T", bound="StreamCredentials")


@_attrs_define
class StreamCredentials:
    """The keys the router acts in the calling app's own Stream app with. Secrets are written and never read back: no
    answer carries one.

        Attributes:
            expected_revision (int): The revision last read, 0 for an app never registered. A write made against an older
                one is a 409.
            keys (list[StreamKeyInput] | None): Every key the router may act in the app with, which replaces those it held.
                Each is checked with Stream. Empty disconnects the app, which needs a proof.
            allow_guests (bool | Unset): Whether guests may be made in the app. Left out keeps what was set, which starts
                off false.
            primary_key (str | Unset): The key tokens are minted with. Left out is the first.
            proof (StreamKeyInput | Unset):
    """

    expected_revision: int
    keys: list[StreamKeyInput] | None
    allow_guests: bool | Unset = UNSET
    primary_key: str | Unset = UNSET
    proof: StreamKeyInput | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        expected_revision = self.expected_revision

        keys: list[dict[str, Any]] | None
        if isinstance(self.keys, list):
            keys = []
            for keys_type_0_item_data in self.keys:
                keys_type_0_item = keys_type_0_item_data.to_dict()
                keys.append(keys_type_0_item)

        else:
            keys = self.keys

        allow_guests = self.allow_guests

        primary_key = self.primary_key

        proof: dict[str, Any] | Unset = UNSET
        if not isinstance(self.proof, Unset):
            proof = self.proof.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "expected_revision": expected_revision,
                "keys": keys,
            }
        )
        if allow_guests is not UNSET:
            field_dict["allow_guests"] = allow_guests
        if primary_key is not UNSET:
            field_dict["primary_key"] = primary_key
        if proof is not UNSET:
            field_dict["proof"] = proof

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.stream_key_input import StreamKeyInput

        d = dict(src_dict)
        expected_revision = d.pop("expected_revision")

        def _parse_keys(data: object) -> list[StreamKeyInput] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                keys_type_0 = []
                _keys_type_0 = data
                for keys_type_0_item_data in _keys_type_0:
                    keys_type_0_item = StreamKeyInput.from_dict(keys_type_0_item_data)

                    keys_type_0.append(keys_type_0_item)

                return keys_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[StreamKeyInput] | None, data)

        keys = _parse_keys(d.pop("keys"))

        allow_guests = d.pop("allow_guests", UNSET)

        primary_key = d.pop("primary_key", UNSET)

        _proof = d.pop("proof", UNSET)
        proof: StreamKeyInput | Unset
        if isinstance(_proof, Unset):
            proof = UNSET
        else:
            proof = StreamKeyInput.from_dict(_proof)

        stream_credentials = cls(
            expected_revision=expected_revision,
            keys=keys,
            allow_guests=allow_guests,
            primary_key=primary_key,
            proof=proof,
        )

        stream_credentials.additional_properties = d
        return stream_credentials

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
