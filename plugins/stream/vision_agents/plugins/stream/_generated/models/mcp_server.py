from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="McpServer")


@_attrs_define
class McpServer:
    """An MCP server the plugin catalog does not have. Every session opens it at the start, with no login, and offers its
    tools to the model; the instructions the server gives are added to the agent's own.

        Attributes:
            name (str): What its tools are prefixed with, as <name>__<tool>. Lowercase, without __, and not a catalog
                plugin's id.
            url (str): Its Streamable HTTP endpoint, over https.
    """

    name: str
    url: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        url = self.url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "url": url,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        url = d.pop("url")

        mcp_server = cls(
            name=name,
            url=url,
        )

        mcp_server.additional_properties = d
        return mcp_server

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
