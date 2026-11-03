from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.campaign_analytics_outcomes import CampaignAnalyticsOutcomes
    from ..models.campaign_dial_day import CampaignDialDay


T = TypeVar("T", bound="CampaignAnalytics")


@_attrs_define
class CampaignAnalytics:
    """
    Attributes:
        connected (int): Contacts who answered at least one call.
        conversion_rate (float): converted / dialed, 0 when nobody was dialed.
        converted (int): Contacts that ended successfully.
        dialed (int): Contacts with at least one call placed.
        dials_per_day (list[CampaignDialDay]): Oldest first, only the days with a dial.
        outcomes (CampaignAnalyticsOutcomes): Contacts that are done, by outcome id, no_response included.
    """

    connected: int
    conversion_rate: float
    converted: int
    dialed: int
    dials_per_day: list[CampaignDialDay]
    outcomes: CampaignAnalyticsOutcomes
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connected = self.connected

        conversion_rate = self.conversion_rate

        converted = self.converted

        dialed = self.dialed

        dials_per_day = []
        for dials_per_day_item_data in self.dials_per_day:
            dials_per_day_item = dials_per_day_item_data.to_dict()
            dials_per_day.append(dials_per_day_item)

        outcomes = self.outcomes.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connected": connected,
                "conversion_rate": conversion_rate,
                "converted": converted,
                "dialed": dialed,
                "dials_per_day": dials_per_day,
                "outcomes": outcomes,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.campaign_analytics_outcomes import CampaignAnalyticsOutcomes
        from ..models.campaign_dial_day import CampaignDialDay

        d = dict(src_dict)
        connected = d.pop("connected")

        conversion_rate = d.pop("conversion_rate")

        converted = d.pop("converted")

        dialed = d.pop("dialed")

        dials_per_day = []
        _dials_per_day = d.pop("dials_per_day")
        for dials_per_day_item_data in _dials_per_day:
            dials_per_day_item = CampaignDialDay.from_dict(dials_per_day_item_data)

            dials_per_day.append(dials_per_day_item)

        outcomes = CampaignAnalyticsOutcomes.from_dict(d.pop("outcomes"))

        campaign_analytics = cls(
            connected=connected,
            conversion_rate=conversion_rate,
            converted=converted,
            dialed=dialed,
            dials_per_day=dials_per_day,
            outcomes=outcomes,
        )

        campaign_analytics.additional_properties = d
        return campaign_analytics

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
