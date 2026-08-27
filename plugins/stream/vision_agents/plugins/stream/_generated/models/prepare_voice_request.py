from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="PrepareVoiceRequest")


@_attrs_define
class PrepareVoiceRequest:
    """
    Attributes:
        providers (list[str] | Unset): Which providers to teach the voice to. Empty means every provider this deployment
            can clone with.
    """

    providers: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if providers is not UNSET:
            field_dict["providers"] = providers

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        providers = cast(list[str], d.pop("providers", UNSET))

        prepare_voice_request = cls(
            providers=providers,
        )

        prepare_voice_request.additional_properties = d
        return prepare_voice_request

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
