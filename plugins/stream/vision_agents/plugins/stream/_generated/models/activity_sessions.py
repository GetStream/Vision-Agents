from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.activity_ending import ActivityEnding
    from ..models.activity_session import ActivitySession


T = TypeVar("T", bound="ActivitySessions")


@_attrs_define
class ActivitySessions:
    """
    Attributes:
        ended (list[ActivityEnding] | None): How many of them ended for each reason, by reason, the ones with no reason
            kept first.
        latest (list[ActivitySession] | None): The 20 opened last, newest first.
        opened (int): How many sessions were opened, running ones included.
        running (int): How many of them have not ended.
    """

    ended: list[ActivityEnding] | None
    latest: list[ActivitySession] | None
    opened: int
    running: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        ended: list[dict[str, Any]] | None
        if isinstance(self.ended, list):
            ended = []
            for ended_type_0_item_data in self.ended:
                ended_type_0_item = ended_type_0_item_data.to_dict()
                ended.append(ended_type_0_item)

        else:
            ended = self.ended

        latest: list[dict[str, Any]] | None
        if isinstance(self.latest, list):
            latest = []
            for latest_type_0_item_data in self.latest:
                latest_type_0_item = latest_type_0_item_data.to_dict()
                latest.append(latest_type_0_item)

        else:
            latest = self.latest

        opened = self.opened

        running = self.running

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "ended": ended,
                "latest": latest,
                "opened": opened,
                "running": running,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.activity_ending import ActivityEnding
        from ..models.activity_session import ActivitySession

        d = dict(src_dict)

        def _parse_ended(data: object) -> list[ActivityEnding] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                ended_type_0 = []
                _ended_type_0 = data
                for ended_type_0_item_data in _ended_type_0:
                    ended_type_0_item = ActivityEnding.from_dict(ended_type_0_item_data)

                    ended_type_0.append(ended_type_0_item)

                return ended_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ActivityEnding] | None, data)

        ended = _parse_ended(d.pop("ended"))

        def _parse_latest(data: object) -> list[ActivitySession] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                latest_type_0 = []
                _latest_type_0 = data
                for latest_type_0_item_data in _latest_type_0:
                    latest_type_0_item = ActivitySession.from_dict(
                        latest_type_0_item_data
                    )

                    latest_type_0.append(latest_type_0_item)

                return latest_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ActivitySession] | None, data)

        latest = _parse_latest(d.pop("latest"))

        opened = d.pop("opened")

        running = d.pop("running")

        activity_sessions = cls(
            ended=ended,
            latest=latest,
            opened=opened,
            running=running,
        )

        activity_sessions.additional_properties = d
        return activity_sessions

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
