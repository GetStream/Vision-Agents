from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="FileSource")


@_attrs_define
class FileSource:
    """A file sent with a question. The router reads it as text once, and the agent answers from it on this turn and every
    turn of the conversation after it.

        Attributes:
            url (str): Public HTTP(S) URL or base64 data URI, such as data:application/pdf;base64,.... A PDF or a Word
                document (DOCX or DOC), told apart by its bytes. At most 20 MiB fetched, or 8 MiB as a data URI. The router
                fetches a URL itself, and refuses one that resolves to a private or loopback address.
            name (str | Unset): What the file is called, shown on the person's message and to the model. Left out, the last
                part of the URL's path, or document for a data URI.
    """

    url: str
    name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        url = self.url

        name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "url": url,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        url = d.pop("url")

        name = d.pop("name", UNSET)

        file_source = cls(
            url=url,
            name=name,
        )

        file_source.additional_properties = d
        return file_source

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
