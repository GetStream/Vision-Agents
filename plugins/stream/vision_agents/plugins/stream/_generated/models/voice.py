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
        created_at (datetime.datetime):
        id (str):
        name (str):
        updated_at (datetime.datetime):
        bindings (list[VoiceBinding] | Unset):
        description (str | Unset):
        samples (list[VoiceSample] | Unset):
    """

    created_at: datetime.datetime
    id: str
    name: str
    updated_at: datetime.datetime
    bindings: list[VoiceBinding] | Unset = UNSET
    description: str | Unset = UNSET
    samples: list[VoiceSample] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        id = self.id

        name = self.name

        updated_at = self.updated_at.isoformat()

        bindings: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.bindings, Unset):
            bindings = []
            for bindings_item_data in self.bindings:
                bindings_item = bindings_item_data.to_dict()
                bindings.append(bindings_item)

        description = self.description

        samples: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.samples, Unset):
            samples = []
            for samples_item_data in self.samples:
                samples_item = samples_item_data.to_dict()
                samples.append(samples_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "id": id,
                "name": name,
                "updated_at": updated_at,
            }
        )
        if bindings is not UNSET:
            field_dict["bindings"] = bindings
        if description is not UNSET:
            field_dict["description"] = description
        if samples is not UNSET:
            field_dict["samples"] = samples

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.voice_binding import VoiceBinding
        from ..models.voice_sample import VoiceSample

        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        name = d.pop("name")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        _bindings = d.pop("bindings", UNSET)
        bindings: list[VoiceBinding] | Unset = UNSET
        if _bindings is not UNSET:
            bindings = []
            for bindings_item_data in _bindings:
                bindings_item = VoiceBinding.from_dict(bindings_item_data)

                bindings.append(bindings_item)

        description = d.pop("description", UNSET)

        _samples = d.pop("samples", UNSET)
        samples: list[VoiceSample] | Unset = UNSET
        if _samples is not UNSET:
            samples = []
            for samples_item_data in _samples:
                samples_item = VoiceSample.from_dict(samples_item_data)

                samples.append(samples_item)

        voice = cls(
            created_at=created_at,
            id=id,
            name=name,
            updated_at=updated_at,
            bindings=bindings,
            description=description,
            samples=samples,
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
