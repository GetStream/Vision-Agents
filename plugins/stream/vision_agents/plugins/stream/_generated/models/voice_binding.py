from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.voice_binding_state import VoiceBindingState
from ..types import UNSET, Unset

T = TypeVar("T", bound="VoiceBinding")


@_attrs_define
class VoiceBinding:
    """
    Attributes:
        provider (str):
        state (VoiceBindingState):
        external_id (str | Unset): What this provider calls the voice.
        error (str | Unset): Why the provider would not take the recordings, when it would not.
        updated_at (datetime.datetime | Unset):
    """

    provider: str
    state: VoiceBindingState
    external_id: str | Unset = UNSET
    error: str | Unset = UNSET
    updated_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        state = self.state.value

        external_id = self.external_id

        error = self.error

        updated_at: str | Unset = UNSET
        if not isinstance(self.updated_at, Unset):
            updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "state": state,
            }
        )
        if external_id is not UNSET:
            field_dict["external_id"] = external_id
        if error is not UNSET:
            field_dict["error"] = error
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        provider = d.pop("provider")

        state = VoiceBindingState(d.pop("state"))

        external_id = d.pop("external_id", UNSET)

        error = d.pop("error", UNSET)

        _updated_at = d.pop("updated_at", UNSET)
        updated_at: datetime.datetime | Unset
        if isinstance(_updated_at, Unset):
            updated_at = UNSET
        else:
            updated_at = datetime.datetime.fromisoformat(_updated_at)

        voice_binding = cls(
            provider=provider,
            state=state,
            external_id=external_id,
            error=error,
            updated_at=updated_at,
        )

        voice_binding.additional_properties = d
        return voice_binding

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
