from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="DirectoryFile")


@_attrs_define
class DirectoryFile:
    """One file of an agent directory, as a sync takes it and an export writes it.

    Attributes:
        content (str): The file, whole, as text.
        path (str): Where the file is in the directory, with forward slashes: agent.yaml, instructions.md, guardrail.md,
            skills/<name>.md or skills/<name>/SKILL.md with the files beside it, knowledge/<file>, knowledge/urls.yaml,
            simulations/<file>.yaml, ui/components.json, ui/blueprints.yaml. Example: skills/refunds/SKILL.md.
    """

    content: str
    path: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        content = self.content

        path = self.path

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "content": content,
                "path": path,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        content = d.pop("content")

        path = d.pop("path")

        directory_file = cls(
            content=content,
            path=path,
        )

        directory_file.additional_properties = d
        return directory_file

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
