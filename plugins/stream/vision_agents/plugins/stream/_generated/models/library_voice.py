from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="LibraryVoice")


@_attrs_define
class LibraryVoice:
    """One voice a provider offers. Everything past the name is what that vendor chose to say about it, in its own words,
    so a field being absent means the vendor did not label it rather than that the voice lacks it.

        Attributes:
            provider (str):
            id (str): What to put in the voice field, in the provider's own terms.
            name (str):
            description (str | Unset):
            gender (str | Unset):
            accent (str | Unset):
            language (str | Unset):
            tags (list[str] | Unset):
            own (bool | Unset): A voice this account made, rather than one from the public library.
            preview (bool | Unset): Whether the voice can be heard.
    """

    provider: str
    id: str
    name: str
    description: str | Unset = UNSET
    gender: str | Unset = UNSET
    accent: str | Unset = UNSET
    language: str | Unset = UNSET
    tags: list[str] | Unset = UNSET
    own: bool | Unset = UNSET
    preview: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider

        id = self.id

        name = self.name

        description = self.description

        gender = self.gender

        accent = self.accent

        language = self.language

        tags: list[str] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags

        own = self.own

        preview = self.preview

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "provider": provider,
                "id": id,
                "name": name,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if gender is not UNSET:
            field_dict["gender"] = gender
        if accent is not UNSET:
            field_dict["accent"] = accent
        if language is not UNSET:
            field_dict["language"] = language
        if tags is not UNSET:
            field_dict["tags"] = tags
        if own is not UNSET:
            field_dict["own"] = own
        if preview is not UNSET:
            field_dict["preview"] = preview

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        provider = d.pop("provider")

        id = d.pop("id")

        name = d.pop("name")

        description = d.pop("description", UNSET)

        gender = d.pop("gender", UNSET)

        accent = d.pop("accent", UNSET)

        language = d.pop("language", UNSET)

        tags = cast(list[str], d.pop("tags", UNSET))

        own = d.pop("own", UNSET)

        preview = d.pop("preview", UNSET)

        library_voice = cls(
            provider=provider,
            id=id,
            name=name,
            description=description,
            gender=gender,
            accent=accent,
            language=language,
            tags=tags,
            own=own,
            preview=preview,
        )

        library_voice.additional_properties = d
        return library_voice

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
