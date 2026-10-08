from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="VideoSource")


@_attrs_define
class VideoSource:
    """
    Attributes:
        url (str): Public HTTPS URL or base64 video data URI, such as data:video/mp4;base64,.... At most 50 MB either
            way. The router fetches a URL itself, and refuses one that resolves to a private or loopback address.
        max_frames (int | Unset): How many frames to sample, evenly spaced across the clip. Default 8.
    """

    url: str
    max_frames: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        url = self.url

        max_frames = self.max_frames

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "url": url,
            }
        )
        if max_frames is not UNSET:
            field_dict["max_frames"] = max_frames

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        url = d.pop("url")

        max_frames = d.pop("max_frames", UNSET)

        video_source = cls(
            url=url,
            max_frames=max_frames,
        )

        video_source.additional_properties = d
        return video_source

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
