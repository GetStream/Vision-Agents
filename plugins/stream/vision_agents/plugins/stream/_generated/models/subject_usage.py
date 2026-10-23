from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.limit_subject import LimitSubject
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.limit_window import LimitWindow
    from ..models.spend_hard_limit_usage import SpendHardLimitUsage


T = TypeVar("T", bound="SubjectUsage")


@_attrs_define
class SubjectUsage:
    """
    Attributes:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an end user of an app, a
            usage category across the organization or within one app, or a team across the organization or within one app. A
            team is the user's team, as its token names it.
        windows (list[LimitWindow] | None): The current window of every interval, daily first.
        api_key (str | Unset):
        app_id (str | Unset):
        category (str | Unset):
        spend_hard_limit (SpendHardLimitUsage | Unset): The most this organization may spend in the UTC day and in the
            calendar month, whatever its credit and its own limits. Set by the operator, never by the customer. A missing
            interval has no limit.
        team (str | Unset):
        user_id (str | Unset):
    """

    subject: LimitSubject
    windows: list[LimitWindow] | None
    api_key: str | Unset = UNSET
    app_id: str | Unset = UNSET
    category: str | Unset = UNSET
    spend_hard_limit: SpendHardLimitUsage | Unset = UNSET
    team: str | Unset = UNSET
    user_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        subject = self.subject.value

        windows: list[dict[str, Any]] | None
        if isinstance(self.windows, list):
            windows = []
            for windows_type_0_item_data in self.windows:
                windows_type_0_item = windows_type_0_item_data.to_dict()
                windows.append(windows_type_0_item)

        else:
            windows = self.windows

        api_key = self.api_key

        app_id = self.app_id

        category = self.category

        spend_hard_limit: dict[str, Any] | Unset = UNSET
        if not isinstance(self.spend_hard_limit, Unset):
            spend_hard_limit = self.spend_hard_limit.to_dict()

        team = self.team

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "subject": subject,
                "windows": windows,
            }
        )
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if app_id is not UNSET:
            field_dict["app_id"] = app_id
        if category is not UNSET:
            field_dict["category"] = category
        if spend_hard_limit is not UNSET:
            field_dict["spend_hard_limit"] = spend_hard_limit
        if team is not UNSET:
            field_dict["team"] = team
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.limit_window import LimitWindow
        from ..models.spend_hard_limit_usage import SpendHardLimitUsage

        d = dict(src_dict)
        subject = LimitSubject(d.pop("subject"))

        def _parse_windows(data: object) -> list[LimitWindow] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                windows_type_0 = []
                _windows_type_0 = data
                for windows_type_0_item_data in _windows_type_0:
                    windows_type_0_item = LimitWindow.from_dict(
                        windows_type_0_item_data
                    )

                    windows_type_0.append(windows_type_0_item)

                return windows_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[LimitWindow] | None, data)

        windows = _parse_windows(d.pop("windows"))

        api_key = d.pop("api_key", UNSET)

        app_id = d.pop("app_id", UNSET)

        category = d.pop("category", UNSET)

        _spend_hard_limit = d.pop("spend_hard_limit", UNSET)
        spend_hard_limit: SpendHardLimitUsage | Unset
        if isinstance(_spend_hard_limit, Unset):
            spend_hard_limit = UNSET
        else:
            spend_hard_limit = SpendHardLimitUsage.from_dict(_spend_hard_limit)

        team = d.pop("team", UNSET)

        user_id = d.pop("user_id", UNSET)

        subject_usage = cls(
            subject=subject,
            windows=windows,
            api_key=api_key,
            app_id=app_id,
            category=category,
            spend_hard_limit=spend_hard_limit,
            team=team,
            user_id=user_id,
        )

        subject_usage.additional_properties = d
        return subject_usage

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
