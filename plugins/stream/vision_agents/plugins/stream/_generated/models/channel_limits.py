from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ChannelLimits")


@_attrs_define
class ChannelLimits:
    """
    Attributes:
        concurrent (int | Unset): How many at once, for a calling channel.
        countries (list[str] | None | Unset): The destination countries allowed, as ISO 3166-1 alpha-2 codes. Absent or
            empty allows any; in overrides, a non-empty list replaces the tier's.
        per_day (int | Unset): How many a UTC day: messages, or minutes for a calling channel.
        recipients_per_day (int | Unset): How many distinct recipients a UTC day.
    """

    concurrent: int | Unset = UNSET
    countries: list[str] | None | Unset = UNSET
    per_day: int | Unset = UNSET
    recipients_per_day: int | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        concurrent = self.concurrent

        countries: list[str] | None | Unset
        if isinstance(self.countries, Unset):
            countries = UNSET
        elif isinstance(self.countries, list):
            countries = self.countries

        else:
            countries = self.countries

        per_day = self.per_day

        recipients_per_day = self.recipients_per_day

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if concurrent is not UNSET:
            field_dict["concurrent"] = concurrent
        if countries is not UNSET:
            field_dict["countries"] = countries
        if per_day is not UNSET:
            field_dict["per_day"] = per_day
        if recipients_per_day is not UNSET:
            field_dict["recipients_per_day"] = recipients_per_day

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        concurrent = d.pop("concurrent", UNSET)

        def _parse_countries(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                countries_type_0 = cast(list[str], data)

                return countries_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        countries = _parse_countries(d.pop("countries", UNSET))

        per_day = d.pop("per_day", UNSET)

        recipients_per_day = d.pop("recipients_per_day", UNSET)

        channel_limits = cls(
            concurrent=concurrent,
            countries=countries,
            per_day=per_day,
            recipients_per_day=recipients_per_day,
        )

        return channel_limits
