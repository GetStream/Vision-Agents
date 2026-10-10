from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.mcp_server_branding import McpServerBranding


T = TypeVar("T", bound="McpServer")


@_attrs_define
class McpServer:
    """An MCP server the plugin catalog does not have. Every session opens it at the start and offers its tools to the
    model; the instructions the server gives are added to the agent's own. It is opened with no login unless it sets
    scopes or user, when it logs in with OAuth as its protected-resource metadata says, registering a client of its own,
    and saving it is refused when the server advertises no such login.

        Attributes:
            name (str): What its tools are prefixed with, as <name>__<tool>. Lowercase, without __, and not a catalog
                plugin's id.
            url (str): Its Streamable HTTP endpoint, over https.
            branding (McpServerBranding | Unset): The serverInfo an MCP server answers initialize with. Every field is
                optional, and a server that sends only its name and version is titled by its name.
            needs_login (bool | Unset): Whether the server requires an OAuth login, as it said when the config was saved:
                protected-resource metadata, or a 401 to a request without a token. Without user, the app logs in once, from the
                dashboard. Absent when it could not be asked, which a session starting asks again.
            scopes (list[str] | Unset): The OAuth scopes its login asks for at consent. Left out, the login asks for the
                scopes_supported the server advertises. Only a server that needs a login may set it. A login made before a
                change keeps what it was granted.
            tools (list[str] | Unset): Offer the model only the server's tools matching these names or path.Match patterns.
                A tool left out is neither listed nor callable. Left out offers every tool.
            user (bool | Unset): Each end user logs in with their own account, in the conversation, the first time the agent
                needs the server, as for a plugin with user, rather than the app once, from the dashboard. Only a server that
                needs a login may set it.
    """

    name: str
    url: str
    branding: McpServerBranding | Unset = UNSET
    needs_login: bool | Unset = UNSET
    scopes: list[str] | Unset = UNSET
    tools: list[str] | Unset = UNSET
    user: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        url = self.url

        branding: dict[str, Any] | Unset = UNSET
        if not isinstance(self.branding, Unset):
            branding = self.branding.to_dict()

        needs_login = self.needs_login

        scopes: list[str] | Unset = UNSET
        if not isinstance(self.scopes, Unset):
            scopes = self.scopes

        tools: list[str] | Unset = UNSET
        if not isinstance(self.tools, Unset):
            tools = self.tools

        user = self.user

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "url": url,
            }
        )
        if branding is not UNSET:
            field_dict["branding"] = branding
        if needs_login is not UNSET:
            field_dict["needs_login"] = needs_login
        if scopes is not UNSET:
            field_dict["scopes"] = scopes
        if tools is not UNSET:
            field_dict["tools"] = tools
        if user is not UNSET:
            field_dict["user"] = user

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.mcp_server_branding import McpServerBranding

        d = dict(src_dict)
        name = d.pop("name")

        url = d.pop("url")

        _branding = d.pop("branding", UNSET)
        branding: McpServerBranding | Unset
        if isinstance(_branding, Unset):
            branding = UNSET
        else:
            branding = McpServerBranding.from_dict(_branding)

        needs_login = d.pop("needs_login", UNSET)

        scopes = cast(list[str], d.pop("scopes", UNSET))

        tools = cast(list[str], d.pop("tools", UNSET))

        user = d.pop("user", UNSET)

        mcp_server = cls(
            name=name,
            url=url,
            branding=branding,
            needs_login=needs_login,
            scopes=scopes,
            tools=tools,
            user=user,
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
