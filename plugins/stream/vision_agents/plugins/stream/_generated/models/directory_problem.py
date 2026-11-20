from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="DirectoryProblem")


@_attrs_define
class DirectoryProblem:
    """One thing wrong with an agent directory, at its place in a file, so an editor can point at it.

    Attributes:
        file (str): The file's path in the directory. Example: agent.yaml.
        message (str):
        column (int | Unset): The column, counted from 1. Left out with line. Example: 3.
        line (int | Unset): The line, counted from 1. Left out when the problem is the file as a whole. Example: 12.
    """

    file: str
    message: str
    column: int | Unset = UNSET
    line: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        file = self.file

        message = self.message

        column = self.column

        line = self.line

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "file": file,
                "message": message,
            }
        )
        if column is not UNSET:
            field_dict["column"] = column
        if line is not UNSET:
            field_dict["line"] = line

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        file = d.pop("file")

        message = d.pop("message")

        column = d.pop("column", UNSET)

        line = d.pop("line", UNSET)

        directory_problem = cls(
            file=file,
            message=message,
            column=column,
            line=line,
        )

        directory_problem.additional_properties = d
        return directory_problem

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
