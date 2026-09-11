from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SessionVideo")


@_attrs_define
class SessionVideo:
    """
    Attributes:
        source (str | Unset): Track or processor source. Omitted requires one unambiguous available source.
        max_frames (int | Unset): Number of recent frames captured for a visual task. Default one.
    """

    source: str | Unset = UNSET
    max_frames: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        source = self.source

        max_frames = self.max_frames

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if source is not UNSET:
            field_dict["source"] = source
        if max_frames is not UNSET:
            field_dict["max_frames"] = max_frames

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        source = d.pop("source", UNSET)

        max_frames = d.pop("max_frames", UNSET)

        session_video = cls(
            source=source,
            max_frames=max_frames,
        )

        session_video.additional_properties = d
        return session_video

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
