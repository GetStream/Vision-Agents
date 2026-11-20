from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.directory_file import DirectoryFile


T = TypeVar("T", bound="AgentDirectory")


@_attrs_define
class AgentDirectory:
    """An agent config as the directory it is synced from: agent.yaml, instructions.md, guardrail.md, its skills, its
    knowledge, its simulations and its UI files.

        Attributes:
            files (list[DirectoryFile]): The directory's files, in path order. Sent back to POST /v1/agents/sync/files, they
                store the agent as it is.
            name (str): What the agent is called, as agent.yaml names it.
            not_exported (list[str]): What this agent holds that its directory cannot say and a sync of these files would
                change, each in a sentence: a connector binding's events, policy or timeout, a connector that grants no tools, a
                knowledge document that is not a text file, UI instructions no directory's files were compiled into. Empty when
                a sync of these files leaves the agent as it is.
    """

    files: list[DirectoryFile]
    name: str
    not_exported: list[str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        files = []
        for files_item_data in self.files:
            files_item = files_item_data.to_dict()
            files.append(files_item)

        name = self.name

        not_exported = self.not_exported

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "files": files,
                "name": name,
                "not_exported": not_exported,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.directory_file import DirectoryFile

        d = dict(src_dict)
        files = []
        _files = d.pop("files")
        for files_item_data in _files:
            files_item = DirectoryFile.from_dict(files_item_data)

            files.append(files_item)

        name = d.pop("name")

        not_exported = cast(list[str], d.pop("not_exported"))

        agent_directory = cls(
            files=files,
            name=name,
            not_exported=not_exported,
        )

        agent_directory.additional_properties = d
        return agent_directory

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
