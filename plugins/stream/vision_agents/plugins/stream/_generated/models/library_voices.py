from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.library_voice import LibraryVoice


T = TypeVar("T", bound="LibraryVoices")


@_attrs_define
class LibraryVoices:
    """
    Attributes:
        voices (list[LibraryVoice]):
        providers (list[str]): The providers that publish a library, sorted by name.
        unavailable (list[str] | Unset): Providers whose library could not be read just now.
    """

    voices: list[LibraryVoice]
    providers: list[str]
    unavailable: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        voices = []
        for voices_item_data in self.voices:
            voices_item = voices_item_data.to_dict()
            voices.append(voices_item)

        providers = self.providers

        unavailable: list[str] | Unset = UNSET
        if not isinstance(self.unavailable, Unset):
            unavailable = self.unavailable

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "voices": voices,
                "providers": providers,
            }
        )
        if unavailable is not UNSET:
            field_dict["unavailable"] = unavailable

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.library_voice import LibraryVoice

        d = dict(src_dict)
        voices = []
        _voices = d.pop("voices")
        for voices_item_data in _voices:
            voices_item = LibraryVoice.from_dict(voices_item_data)

            voices.append(voices_item)

        providers = cast(list[str], d.pop("providers"))

        unavailable = cast(list[str], d.pop("unavailable", UNSET))

        library_voices = cls(
            voices=voices,
            providers=providers,
            unavailable=unavailable,
        )

        library_voices.additional_properties = d
        return library_voices

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
