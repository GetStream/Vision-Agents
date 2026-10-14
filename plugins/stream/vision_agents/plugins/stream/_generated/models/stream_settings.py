from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.stream_tenancy import StreamTenancy
from ..models.stream_type_state import StreamTypeState
from ..models.stream_writes_into import StreamWritesInto
from ..types import UNSET, Unset

T = TypeVar("T", bound="StreamSettings")


@_attrs_define
class StreamSettings:
    """Which Stream app the router writes the calling app's conversations, transcripts, calls and phone lines into, and
    whether that app holds the types they need.

        Attributes:
            call_type (StreamTypeState): Whether a Stream app holds a type the router needs. unknown is a type Stream could
                not be asked about.
            channel_type (StreamTypeState): Whether a Stream app holds a type the router needs. unknown is a type Stream
                could not be asked about.
            tenancy (StreamTenancy): Whose Stream app the router acts in. deployment is one app, the router's own, for every
                app it serves; app is each app's own.
            writes_into (StreamWritesInto): Which Stream app the calling app's work is written into: this_app is its own,
                deployment_app is the router's own app, shared with every app it serves that has none, and nowhere is no app at
                all, so conversations are not kept and calls cannot be made.
            checked_at (datetime.datetime | Unset): When Stream was asked. Absent when it could not be, and the types are
                then unknown. Answers are reused for a minute.
            stream_app_id (int | Unset): The calling app's own Stream app id, in app mode, when its work is written there.
    """

    call_type: StreamTypeState
    channel_type: StreamTypeState
    tenancy: StreamTenancy
    writes_into: StreamWritesInto
    checked_at: datetime.datetime | Unset = UNSET
    stream_app_id: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        call_type = self.call_type.value

        channel_type = self.channel_type.value

        tenancy = self.tenancy.value

        writes_into = self.writes_into.value

        checked_at: str | Unset = UNSET
        if not isinstance(self.checked_at, Unset):
            checked_at = self.checked_at.isoformat()

        stream_app_id = self.stream_app_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "call_type": call_type,
                "channel_type": channel_type,
                "tenancy": tenancy,
                "writes_into": writes_into,
            }
        )
        if checked_at is not UNSET:
            field_dict["checked_at"] = checked_at
        if stream_app_id is not UNSET:
            field_dict["stream_app_id"] = stream_app_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        call_type = StreamTypeState(d.pop("call_type"))

        channel_type = StreamTypeState(d.pop("channel_type"))

        tenancy = StreamTenancy(d.pop("tenancy"))

        writes_into = StreamWritesInto(d.pop("writes_into"))

        _checked_at = d.pop("checked_at", UNSET)
        checked_at: datetime.datetime | Unset
        if isinstance(_checked_at, Unset):
            checked_at = UNSET
        else:
            checked_at = datetime.datetime.fromisoformat(_checked_at)

        stream_app_id = d.pop("stream_app_id", UNSET)

        stream_settings = cls(
            call_type=call_type,
            channel_type=channel_type,
            tenancy=tenancy,
            writes_into=writes_into,
            checked_at=checked_at,
            stream_app_id=stream_app_id,
        )

        stream_settings.additional_properties = d
        return stream_settings

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
