from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="KnowledgeUrlDeclaration")


@_attrs_define
class KnowledgeUrlDeclaration:
    """A page an agent directory declares, in the knowledge base named after it.

    Attributes:
        url (str):  Example: https://example.com/pricing.
        description (str | Unset):
        refresh_hours (int | Unset): How often the page is read again on its own, in hours. Omit it and the page is read
            on every sync that changes the directory, never on a schedule. Example: 24.
        title (str | Unset):  Example: Pricing.
    """

    url: str
    description: str | Unset = UNSET
    refresh_hours: int | Unset = UNSET
    title: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        url = self.url

        description = self.description

        refresh_hours = self.refresh_hours

        title = self.title

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "url": url,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if refresh_hours is not UNSET:
            field_dict["refresh_hours"] = refresh_hours
        if title is not UNSET:
            field_dict["title"] = title

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        url = d.pop("url")

        description = d.pop("description", UNSET)

        refresh_hours = d.pop("refresh_hours", UNSET)

        title = d.pop("title", UNSET)

        knowledge_url_declaration = cls(
            url=url,
            description=description,
            refresh_hours=refresh_hours,
            title=title,
        )

        knowledge_url_declaration.additional_properties = d
        return knowledge_url_declaration

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
