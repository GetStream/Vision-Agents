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
        error (str | Unset): Why the provider would not take the recordings, when it would not.
        external_id (str | Unset): What this provider calls the voice.
        synced_at (datetime.datetime | Unset): When this provider last came back with a voice that can be spoken in,
            absent until one does. It is not updated_at, which moves again when a binding goes back to pending.
        updated_at (datetime.datetime | Unset):
    """

    provider: str
    state: VoiceBindingState
    error: str | Unset = UNSET
    external_id: str | Unset = UNSET
    synced_at: datetime.datetime | Unset = UNSET
    updated_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        state = self.state.value

        error = self.error

        external_id = self.external_id

        synced_at: str | Unset = UNSET
        if not isinstance(self.synced_at, Unset):
            synced_at = self.synced_at.isoformat()

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
        if error is not UNSET:
            field_dict["error"] = error
        if external_id is not UNSET:
            field_dict["external_id"] = external_id
        if synced_at is not UNSET:
            field_dict["synced_at"] = synced_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        provider = d.pop("provider")

        state = VoiceBindingState(d.pop("state"))

        error = d.pop("error", UNSET)

        external_id = d.pop("external_id", UNSET)

        _synced_at = d.pop("synced_at", UNSET)
        synced_at: datetime.datetime | Unset
        if isinstance(_synced_at, Unset):
            synced_at = UNSET
        else:
            synced_at = datetime.datetime.fromisoformat(_synced_at)

        _updated_at = d.pop("updated_at", UNSET)
        updated_at: datetime.datetime | Unset
        if isinstance(_updated_at, Unset):
            updated_at = UNSET
        else:
            updated_at = datetime.datetime.fromisoformat(_updated_at)

        voice_binding = cls(
            provider=provider,
            state=state,
            error=error,
            external_id=external_id,
            synced_at=synced_at,
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
