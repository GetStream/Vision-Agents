from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="SkillRequestFiles")


@_attrs_define
class SkillRequestFiles:
    """The rest of the skill's folder beside SKILL.md, each file's text by its path relative to the folder, such as
    reference/forms.md or scripts/fill.py. The model reads them one at a time when it needs them. A file that is not
    .md, .mdx or .txt is a script, run in the sandbox, so a skill with one belongs only to an agent with a sandbox. At
    most 64 files and 1 MB in all.

    """

    additional_properties: dict[str, str] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        skill_request_files = cls()

        skill_request_files.additional_properties = d
        return skill_request_files

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> str:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
