from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.voice_binding import VoiceBinding
    from ..models.voice_sample import VoiceSample


T = TypeVar("T", bound="Voice")


@_attrs_define
class Voice:
    """
    Attributes:
        id (str):
        name (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        description (str | Unset):
        samples (list[VoiceSample] | Unset):
        bindings (list[VoiceBinding] | Unset):
    """

    id: str
    name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    description: str | Unset = UNSET
    samples: list[VoiceSample] | Unset = UNSET
    bindings: list[VoiceBinding] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        description = self.description

        samples: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.samples, Unset):
            samples = []
            for samples_item_data in self.samples:
                samples_item = samples_item_data.to_dict()
                samples.append(samples_item)

        bindings: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.bindings, Unset):
            bindings = []
            for bindings_item_data in self.bindings:
                bindings_item = bindings_item_data.to_dict()
                bindings.append(bindings_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if samples is not UNSET:
            field_dict["samples"] = samples
        if bindings is not UNSET:
            field_dict["bindings"] = bindings

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.voice_binding import VoiceBinding
        from ..models.voice_sample import VoiceSample

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        description = d.pop("description", UNSET)

        _samples = d.pop("samples", UNSET)
        samples: list[VoiceSample] | Unset = UNSET
        if _samples is not UNSET:
            samples = []
            for samples_item_data in _samples:
                samples_item = VoiceSample.from_dict(samples_item_data)

                samples.append(samples_item)

        _bindings = d.pop("bindings", UNSET)
        bindings: list[VoiceBinding] | Unset = UNSET
        if _bindings is not UNSET:
            bindings = []
            for bindings_item_data in _bindings:
                bindings_item = VoiceBinding.from_dict(bindings_item_data)

                bindings.append(bindings_item)

        voice = cls(
            id=id,
            name=name,
            created_at=created_at,
            updated_at=updated_at,
            description=description,
            samples=samples,
            bindings=bindings,
        )

        voice.additional_properties = d
        return voice

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
