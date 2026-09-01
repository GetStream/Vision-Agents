from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AuthorizePluginRequest")


@_attrs_define
class AuthorizePluginRequest:
    """
    Attributes:
        instance_url (str | Unset): The shop hostname or Salesforce my-domain. Required for plugins that have no single
            global URL.
    """

    instance_url: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instance_url = self.instance_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if instance_url is not UNSET:
            field_dict["instance_url"] = instance_url

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        instance_url = d.pop("instance_url", UNSET)

        authorize_plugin_request = cls(
            instance_url=instance_url,
        )

        authorize_plugin_request.additional_properties = d
        return authorize_plugin_request

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
