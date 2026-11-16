from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.calling_hours_windows import CallingHoursWindows
    from ..models.campaign_time_zone import CampaignTimeZone


T = TypeVar("T", bound="CallingHours")


@_attrs_define
class CallingHours:
    """
    Attributes:
        excluded_dates (list[str]): YYYY-MM-DD, the contact's local date, on which nobody is rung.
        time_zone (CampaignTimeZone):
        windows (CallingHoursWindows): Keyed monday to sunday. A day left out rings nobody.
    """

    excluded_dates: list[str]
    time_zone: CampaignTimeZone
    windows: CallingHoursWindows
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        excluded_dates = self.excluded_dates

        time_zone = self.time_zone.to_dict()

        windows = self.windows.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "excluded_dates": excluded_dates,
                "time_zone": time_zone,
                "windows": windows,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.calling_hours_windows import CallingHoursWindows
        from ..models.campaign_time_zone import CampaignTimeZone

        d = dict(src_dict)
        excluded_dates = cast(list[str], d.pop("excluded_dates"))

        time_zone = CampaignTimeZone.from_dict(d.pop("time_zone"))

        windows = CallingHoursWindows.from_dict(d.pop("windows"))

        calling_hours = cls(
            excluded_dates=excluded_dates,
            time_zone=time_zone,
            windows=windows,
        )

        calling_hours.additional_properties = d
        return calling_hours

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
