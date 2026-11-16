from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.response_branch_of import ResponseBranchOf

T = TypeVar("T", bound="ResponseBranch")


@_attrs_define
class ResponseBranch:
    """
    Attributes:
        count (int): How many there are.
        index (int): Its place, from 1, oldest first.
        of (ResponseBranchOf): reply counts the answers to one question, a regenerate's; question the questions asked at
            one point, an edit's.
    """

    count: int
    index: int
    of: ResponseBranchOf
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        count = self.count

        index = self.index

        of = self.of.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "count": count,
                "index": index,
                "of": of,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        count = d.pop("count")

        index = d.pop("index")

        of = ResponseBranchOf._decode(d.pop("of"))

        response_branch = cls(
            count=count,
            index=index,
            of=of,
        )

        response_branch.additional_properties = d
        return response_branch

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
