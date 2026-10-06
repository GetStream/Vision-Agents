from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.stream_app_state import StreamAppState
from ..models.stream_tenancy import StreamTenancy
from ..models.stream_type_state import StreamTypeState
from ..models.stream_writes_into import StreamWritesInto
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.stream_key_state import StreamKeyState


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
            allow_guests (bool | Unset): Whether guests may be made in the registered app.
            checked_at (datetime.datetime | Unset): When Stream was asked. Absent when it could not be, and the types are
                then unknown. Answers are reused for a minute.
            keys (list[StreamKeyState] | None | Unset): The registered app's keys, oldest first. No secret is ever read
                back.
            primary_key (str | Unset): The key tokens are minted with.
            revision (int | Unset): The registration's revision, 0 for an app that registered none. A write names the one it
                read.
            state (StreamAppState | Unset): Whether the router acts in a registered app. disconnected is one the app took
                back, and blocked one Stream suspended or that stopped checking tokens. Neither is ever written into the
                router's own app instead.
            state_reason (str | Unset): Why the router stopped acting in the app, for one that is blocked.
            stream_app_id (int | Unset): The registered app's own id.
    """

    call_type: StreamTypeState
    channel_type: StreamTypeState
    tenancy: StreamTenancy
    writes_into: StreamWritesInto
    allow_guests: bool | Unset = UNSET
    checked_at: datetime.datetime | Unset = UNSET
    keys: list[StreamKeyState] | None | Unset = UNSET
    primary_key: str | Unset = UNSET
    revision: int | Unset = UNSET
    state: StreamAppState | Unset = UNSET
    state_reason: str | Unset = UNSET
    stream_app_id: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        call_type = self.call_type.value

        channel_type = self.channel_type.value

        tenancy = self.tenancy.value

        writes_into = self.writes_into.value

        allow_guests = self.allow_guests

        checked_at: str | Unset = UNSET
        if not isinstance(self.checked_at, Unset):
            checked_at = self.checked_at.isoformat()

        keys: list[dict[str, Any]] | None | Unset
        if isinstance(self.keys, Unset):
            keys = UNSET
        elif isinstance(self.keys, list):
            keys = []
            for keys_type_0_item_data in self.keys:
                keys_type_0_item = keys_type_0_item_data.to_dict()
                keys.append(keys_type_0_item)

        else:
            keys = self.keys

        primary_key = self.primary_key

        revision = self.revision

        state: str | Unset = UNSET
        if not isinstance(self.state, Unset):
            state = self.state.value

        state_reason = self.state_reason

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
        if allow_guests is not UNSET:
            field_dict["allow_guests"] = allow_guests
        if checked_at is not UNSET:
            field_dict["checked_at"] = checked_at
        if keys is not UNSET:
            field_dict["keys"] = keys
        if primary_key is not UNSET:
            field_dict["primary_key"] = primary_key
        if revision is not UNSET:
            field_dict["revision"] = revision
        if state is not UNSET:
            field_dict["state"] = state
        if state_reason is not UNSET:
            field_dict["state_reason"] = state_reason
        if stream_app_id is not UNSET:
            field_dict["stream_app_id"] = stream_app_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.stream_key_state import StreamKeyState

        d = dict(src_dict)
        call_type = StreamTypeState(d.pop("call_type"))

        channel_type = StreamTypeState(d.pop("channel_type"))

        tenancy = StreamTenancy(d.pop("tenancy"))

        writes_into = StreamWritesInto(d.pop("writes_into"))

        allow_guests = d.pop("allow_guests", UNSET)

        _checked_at = d.pop("checked_at", UNSET)
        checked_at: datetime.datetime | Unset
        if isinstance(_checked_at, Unset):
            checked_at = UNSET
        else:
            checked_at = datetime.datetime.fromisoformat(_checked_at)

        def _parse_keys(data: object) -> list[StreamKeyState] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                keys_type_0 = []
                _keys_type_0 = data
                for keys_type_0_item_data in _keys_type_0:
                    keys_type_0_item = StreamKeyState.from_dict(keys_type_0_item_data)

                    keys_type_0.append(keys_type_0_item)

                return keys_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[StreamKeyState] | None | Unset, data)

        keys = _parse_keys(d.pop("keys", UNSET))

        primary_key = d.pop("primary_key", UNSET)

        revision = d.pop("revision", UNSET)

        _state = d.pop("state", UNSET)
        state: StreamAppState | Unset
        if isinstance(_state, Unset):
            state = UNSET
        else:
            state = StreamAppState(_state)

        state_reason = d.pop("state_reason", UNSET)

        stream_app_id = d.pop("stream_app_id", UNSET)

        stream_settings = cls(
            call_type=call_type,
            channel_type=channel_type,
            tenancy=tenancy,
            writes_into=writes_into,
            allow_guests=allow_guests,
            checked_at=checked_at,
            keys=keys,
            primary_key=primary_key,
            revision=revision,
            state=state,
            state_reason=state_reason,
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
