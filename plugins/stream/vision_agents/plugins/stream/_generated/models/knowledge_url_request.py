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
        namespace (str): The knowledge base to fill, which is what a config's knowledge_namespace names.
             Example: docs.
        url (str): The page to read. It must be http or https: this is handed to a crawler and then used to key the
            passages it becomes.
             Example: https://example.com/pricing.
        title (str | Unset): What to call the page, for a reader of the subscription. Optional: a page that is not named
            here is named by what it called itself when it was last read.
             Example: Pricing.
        description (str | Unset): What the page is, for a reader of the subscription. Optional, and kept as written: it
            says why this page is subscribed to, which a crawler cannot know.
             Example: What each plan includes and where the limits are..
    """

    namespace: str
    url: str
    title: str | Unset = UNSET
    description: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        namespace = self.namespace

        url = self.url

        title = self.title

        description = self.description

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "namespace": namespace,
                "url": url,
            }
        )
        if title is not UNSET:
            field_dict["title"] = title
        if description is not UNSET:
            field_dict["description"] = description

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        namespace = d.pop("namespace")

        url = d.pop("url")

        title = d.pop("title", UNSET)

        description = d.pop("description", UNSET)

        knowledge_url_request = cls(
            namespace=namespace,
            url=url,
            title=title,
            description=description,
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
