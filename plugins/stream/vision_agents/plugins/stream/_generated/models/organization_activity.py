from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.activity_step import ActivityStep
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.activity_area import ActivityArea
    from ..models.activity_latency import ActivityLatency
    from ..models.activity_meter import ActivityMeter
    from ..models.activity_sessions import ActivitySessions


T = TypeVar("T", bound="OrganizationActivity")


@_attrs_define
class OrganizationActivity:
    """
    Attributes:
        areas (list[ActivityArea] | None): Every area and client the organization's apps made requests in, by area then
            client.
        furthest_step (ActivityStep): The furthest an organization got: none, dashboard_reads, agent_setup,
            conversations, then steady_use, which is conversations or model calls answered on three UTC days or more. A step
            counts only answered requests: an area whose every request was refused was tried, not reached.
        meters (list[ActivityMeter] | None): What the organization's apps' work came to, by modality, from the shards
            listed in shards.
        organization_id (str): The organization.
        since (datetime.datetime): The start of what was read. Areas are kept by UTC day, so they count the whole day it
            falls on.
        shards (list[str] | None): The names of the shards whose requests the meters sum, this one's first. A deployment
            that sets no postgres.shards lists only its own shard, named by a digest, and its meters cover that shard alone.
            Areas are read from the database every shard shares and cover all of them.
        latency (list[ActivityLatency] | None): How long the turns of the organization's apps took, every agent summed,
            one item per leg: stt_latency, llm_ttft, tts_ttfb, then roundtrip. Turns are kept by the hour they started in,
            so they count the whole hour since falls in. The percentiles are read off bins 5% wide, so they are within 5% of
            the true ones.
        sessions (ActivitySessions):
        first_seen (datetime.datetime | Unset): When the first request of any area arrived, absent when none did.
        last_seen (datetime.datetime | Unset): When the last request of any area arrived, absent when none did.
    """

    areas: list[ActivityArea] | None
    furthest_step: ActivityStep
    meters: list[ActivityMeter] | None
    organization_id: str
    since: datetime.datetime
    shards: list[str] | None
    latency: list[ActivityLatency] | None
    sessions: ActivitySessions
    first_seen: datetime.datetime | Unset = UNSET
    last_seen: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        areas: list[dict[str, Any]] | None
        if isinstance(self.areas, list):
            areas = []
            for areas_type_0_item_data in self.areas:
                areas_type_0_item = areas_type_0_item_data.to_dict()
                areas.append(areas_type_0_item)

        else:
            areas = self.areas

        furthest_step = self.furthest_step.value

        meters: list[dict[str, Any]] | None
        if isinstance(self.meters, list):
            meters = []
            for meters_type_0_item_data in self.meters:
                meters_type_0_item = meters_type_0_item_data.to_dict()
                meters.append(meters_type_0_item)

        else:
            meters = self.meters

        organization_id = self.organization_id

        since = self.since.isoformat()

        shards: list[str] | None
        if isinstance(self.shards, list):
            shards = self.shards

        else:
            shards = self.shards

        latency: list[dict[str, Any]] | None
        if isinstance(self.latency, list):
            latency = []
            for latency_type_0_item_data in self.latency:
                latency_type_0_item = latency_type_0_item_data.to_dict()
                latency.append(latency_type_0_item)

        else:
            latency = self.latency

        sessions = self.sessions.to_dict()

        first_seen: str | Unset = UNSET
        if not isinstance(self.first_seen, Unset):
            first_seen = self.first_seen.isoformat()

        last_seen: str | Unset = UNSET
        if not isinstance(self.last_seen, Unset):
            last_seen = self.last_seen.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "areas": areas,
                "furthest_step": furthest_step,
                "meters": meters,
                "organization_id": organization_id,
                "since": since,
                "shards": shards,
                "latency": latency,
                "sessions": sessions,
            }
        )
        if first_seen is not UNSET:
            field_dict["first_seen"] = first_seen
        if last_seen is not UNSET:
            field_dict["last_seen"] = last_seen

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.activity_area import ActivityArea
        from ..models.activity_latency import ActivityLatency
        from ..models.activity_meter import ActivityMeter
        from ..models.activity_sessions import ActivitySessions

        d = dict(src_dict)

        def _parse_areas(data: object) -> list[ActivityArea] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                areas_type_0 = []
                _areas_type_0 = data
                for areas_type_0_item_data in _areas_type_0:
                    areas_type_0_item = ActivityArea.from_dict(areas_type_0_item_data)

                    areas_type_0.append(areas_type_0_item)

                return areas_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ActivityArea] | None, data)

        areas = _parse_areas(d.pop("areas"))

        furthest_step = ActivityStep._decode(d.pop("furthest_step"))

        def _parse_meters(data: object) -> list[ActivityMeter] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                meters_type_0 = []
                _meters_type_0 = data
                for meters_type_0_item_data in _meters_type_0:
                    meters_type_0_item = ActivityMeter.from_dict(
                        meters_type_0_item_data
                    )

                    meters_type_0.append(meters_type_0_item)

                return meters_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ActivityMeter] | None, data)

        meters = _parse_meters(d.pop("meters"))

        organization_id = d.pop("organization_id")

        since = datetime.datetime.fromisoformat(d.pop("since"))

        def _parse_shards(data: object) -> list[str] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                shards_type_0 = cast(list[str], data)

                return shards_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None, data)

        shards = _parse_shards(d.pop("shards"))

        def _parse_latency(data: object) -> list[ActivityLatency] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                latency_type_0 = []
                _latency_type_0 = data
                for latency_type_0_item_data in _latency_type_0:
                    latency_type_0_item = ActivityLatency.from_dict(
                        latency_type_0_item_data
                    )

                    latency_type_0.append(latency_type_0_item)

                return latency_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ActivityLatency] | None, data)

        latency = _parse_latency(d.pop("latency"))

        sessions = ActivitySessions.from_dict(d.pop("sessions"))

        _first_seen = d.pop("first_seen", UNSET)
        first_seen: datetime.datetime | Unset
        if isinstance(_first_seen, Unset):
            first_seen = UNSET
        else:
            first_seen = datetime.datetime.fromisoformat(_first_seen)

        _last_seen = d.pop("last_seen", UNSET)
        last_seen: datetime.datetime | Unset
        if isinstance(_last_seen, Unset):
            last_seen = UNSET
        else:
            last_seen = datetime.datetime.fromisoformat(_last_seen)

        organization_activity = cls(
            areas=areas,
            furthest_step=furthest_step,
            meters=meters,
            organization_id=organization_id,
            since=since,
            shards=shards,
            latency=latency,
            sessions=sessions,
            first_seen=first_seen,
            last_seen=last_seen,
        )

        organization_activity.additional_properties = d
        return organization_activity

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
