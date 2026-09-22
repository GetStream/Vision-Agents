from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.candidate import Candidate


T = TypeVar("T", bound="Route")


@_attrs_define
class Route:
    """
    Attributes:
        id (str): The shortcut, which is what a config or request names as its target. Example: llm-fast.
        title (str):  Example: Fast conversational.
        description (str):
        candidates (list[Candidate]): The models the shortcut resolves to right now, best first.
    """

    id: str
    title: str
    description: str
    candidates: list[Candidate]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        title = self.title

        description = self.description

        candidates = []
        for candidates_item_data in self.candidates:
            candidates_item = candidates_item_data.to_dict()
            candidates.append(candidates_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "title": title,
                "description": description,
                "candidates": candidates,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.candidate import Candidate

        d = dict(src_dict)
        id = d.pop("id")

        title = d.pop("title")

        description = d.pop("description")

        candidates = []
        _candidates = d.pop("candidates")
        for candidates_item_data in _candidates:
            candidates_item = Candidate.from_dict(candidates_item_data)

            candidates.append(candidates_item)

        route = cls(
            id=id,
            title=title,
            description=description,
            candidates=candidates,
        )

        route.additional_properties = d
        return route

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
