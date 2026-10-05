from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="McpServerBranding")


@_attrs_define
class McpServerBranding:
    """The serverInfo an MCP server answers initialize with. Every field is optional, and a server that sends only its name
    and version is titled by its name.

        Attributes:
            description (str | Unset):
            icon_url (str | Unset): Its first icon served over https, as the server links it.
            title (str | Unset): Its display title, or its name when it gives none.
            version (str | Unset):
            website_url (str | Unset):
    """

    description: str | Unset = UNSET
    icon_url: str | Unset = UNSET
    title: str | Unset = UNSET
    version: str | Unset = UNSET
    website_url: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        icon_url = self.icon_url

        title = self.title

        version = self.version

        website_url = self.website_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if description is not UNSET:
            field_dict["description"] = description
        if icon_url is not UNSET:
            field_dict["icon_url"] = icon_url
        if title is not UNSET:
            field_dict["title"] = title
        if version is not UNSET:
            field_dict["version"] = version
        if website_url is not UNSET:
            field_dict["website_url"] = website_url

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        description = d.pop("description", UNSET)

        icon_url = d.pop("icon_url", UNSET)

        title = d.pop("title", UNSET)

        version = d.pop("version", UNSET)

        website_url = d.pop("website_url", UNSET)

        mcp_server_branding = cls(
            description=description,
            icon_url=icon_url,
            title=title,
            version=version,
            website_url=website_url,
        )

        mcp_server_branding.additional_properties = d
        return mcp_server_branding

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
