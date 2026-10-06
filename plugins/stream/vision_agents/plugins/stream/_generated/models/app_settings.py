from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.stream_settings import StreamSettings


T = TypeVar("T", bound="AppSettings")


@_attrs_define
class AppSettings:
    """What the router does for the calling app. It never carries a secret.

    Attributes:
        stream (StreamSettings): Which Stream app the router writes the calling app's conversations, transcripts, calls
            and phone lines into, and whether that app holds the types they need.
    """

    stream: StreamSettings
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        stream = self.stream.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "stream": stream,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.stream_settings import StreamSettings

        d = dict(src_dict)
        stream = StreamSettings.from_dict(d.pop("stream"))

        app_settings = cls(
            stream=stream,
        )

        app_settings.additional_properties = d
        return app_settings

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
