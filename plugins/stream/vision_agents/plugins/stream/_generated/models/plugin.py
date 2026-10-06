from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.plugin_setup_step import PluginSetupStep


T = TypeVar("T", bound="Plugin")


@_attrs_define
class Plugin:
    """One hosted MCP server from the built-in catalog.

    Attributes:
        category (str):
        description (str):
        id (str):
        logo_url (str): Where this deployment serves the plugin's logo, as an SVG needing no credential.
        name (str):
        client_required (bool | Unset): True when the provider registers no client on the fly, so a config needs one of
            the app's own, set with setPluginClient, before anybody can connect the plugin.
        instance_hint (str | Unset):
        instance_required (bool | Unset):
        readonly (bool | Unset): True when the plugin has a read-only endpoint an agent may pick on its entry.
        redirect_uri (str | Unset): The redirect URI that client has to list, which is this deployment's. Only with
            client_required.
        scopes_supported (list[str] | Unset): The OAuth scopes an agent may ask for on its entry, as the server
            advertises them. Absent when the server says nothing, and any scope is then passed through.
        setup_steps (list[PluginSetupStep] | Unset): What to do there, in order, before pasting the client into
            setPluginClient. Absent when the catalog has no instructions for the plugin.
        setup_url (str | Unset): Where the app creates that client with the provider. Only with client_required.
        toolsets (list[str] | Unset): The groups of tools an agent may limit the plugin to on its entry. Absent when it
            cannot be limited.
    """

    category: str
    description: str
    id: str
    logo_url: str
    name: str
    client_required: bool | Unset = UNSET
    instance_hint: str | Unset = UNSET
    instance_required: bool | Unset = UNSET
    readonly: bool | Unset = UNSET
    redirect_uri: str | Unset = UNSET
    scopes_supported: list[str] | Unset = UNSET
    setup_steps: list[PluginSetupStep] | Unset = UNSET
    setup_url: str | Unset = UNSET
    toolsets: list[str] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        category = self.category

        description = self.description

        id = self.id

        logo_url = self.logo_url

        name = self.name

        client_required = self.client_required

        instance_hint = self.instance_hint

        instance_required = self.instance_required

        readonly = self.readonly

        redirect_uri = self.redirect_uri

        scopes_supported: list[str] | Unset = UNSET
        if not isinstance(self.scopes_supported, Unset):
            scopes_supported = self.scopes_supported

        setup_steps: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.setup_steps, Unset):
            setup_steps = []
            for setup_steps_item_data in self.setup_steps:
                setup_steps_item = setup_steps_item_data.to_dict()
                setup_steps.append(setup_steps_item)

        setup_url = self.setup_url

        toolsets: list[str] | Unset = UNSET
        if not isinstance(self.toolsets, Unset):
            toolsets = self.toolsets

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "category": category,
                "description": description,
                "id": id,
                "logo_url": logo_url,
                "name": name,
            }
        )
        if client_required is not UNSET:
            field_dict["client_required"] = client_required
        if instance_hint is not UNSET:
            field_dict["instance_hint"] = instance_hint
        if instance_required is not UNSET:
            field_dict["instance_required"] = instance_required
        if readonly is not UNSET:
            field_dict["readonly"] = readonly
        if redirect_uri is not UNSET:
            field_dict["redirect_uri"] = redirect_uri
        if scopes_supported is not UNSET:
            field_dict["scopes_supported"] = scopes_supported
        if setup_steps is not UNSET:
            field_dict["setup_steps"] = setup_steps
        if setup_url is not UNSET:
            field_dict["setup_url"] = setup_url
        if toolsets is not UNSET:
            field_dict["toolsets"] = toolsets

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.plugin_setup_step import PluginSetupStep

        d = dict(src_dict)
        category = d.pop("category")

        description = d.pop("description")

        id = d.pop("id")

        logo_url = d.pop("logo_url")

        name = d.pop("name")

        client_required = d.pop("client_required", UNSET)

        instance_hint = d.pop("instance_hint", UNSET)

        instance_required = d.pop("instance_required", UNSET)

        readonly = d.pop("readonly", UNSET)

        redirect_uri = d.pop("redirect_uri", UNSET)

        scopes_supported = cast(list[str], d.pop("scopes_supported", UNSET))

        _setup_steps = d.pop("setup_steps", UNSET)
        setup_steps: list[PluginSetupStep] | Unset = UNSET
        if _setup_steps is not UNSET:
            setup_steps = []
            for setup_steps_item_data in _setup_steps:
                setup_steps_item = PluginSetupStep.from_dict(setup_steps_item_data)

                setup_steps.append(setup_steps_item)

        setup_url = d.pop("setup_url", UNSET)

        toolsets = cast(list[str], d.pop("toolsets", UNSET))

        plugin = cls(
            category=category,
            description=description,
            id=id,
            logo_url=logo_url,
            name=name,
            client_required=client_required,
            instance_hint=instance_hint,
            instance_required=instance_required,
            readonly=readonly,
            redirect_uri=redirect_uri,
            scopes_supported=scopes_supported,
            setup_steps=setup_steps,
            setup_url=setup_url,
            toolsets=toolsets,
        )

        plugin.additional_properties = d
        return plugin

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
