from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="KnowledgeUrlRequest")


@_attrs_define
class KnowledgeUrlRequest:
    """
    Attributes:
        namespace (str): The knowledge base to fill, which is what a config's knowledge_namespace names. Example: docs.
        url (str): The page to read. It must be http or https: this is handed to a crawler and then used to key the
            passages it becomes. Example: https://example.com/pricing.
        description (str | Unset): What the page is, for a reader of the subscription. Optional, and kept as written: it
            says why this page is subscribed to, which a crawler cannot know. Example: What each plan includes and where the
            limits are..
        refresh_hours (int | Unset): How often the page is read again on its own, in hours. Omit it, or send zero, and
            the page is read when it is added and when it is re-indexed, never on a schedule. Adding the page again replaces
            it. Example: 24.
        title (str | Unset): What to call the page, for a reader of the subscription. Optional: a page that is not named
            here is named by what it called itself when it was last read. Example: Pricing.
    """

    namespace: str
    url: str
    description: str | Unset = UNSET
    refresh_hours: int | Unset = UNSET
    title: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        namespace = self.namespace

        url = self.url

        description = self.description

        refresh_hours = self.refresh_hours

        title = self.title

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "namespace": namespace,
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
        namespace = d.pop("namespace")

        url = d.pop("url")

        description = d.pop("description", UNSET)

        refresh_hours = d.pop("refresh_hours", UNSET)

        title = d.pop("title", UNSET)

        knowledge_url_request = cls(
            namespace=namespace,
            url=url,
            description=description,
            refresh_hours=refresh_hours,
            title=title,
        )

        knowledge_url_request.additional_properties = d
        return knowledge_url_request

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
