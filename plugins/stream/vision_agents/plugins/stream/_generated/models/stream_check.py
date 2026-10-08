from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.app_settings import AppSettings


T = TypeVar("T", bound="StreamCheck")


@_attrs_define
class StreamCheck:
    """
    Attributes:
        reattach (list[str] | None):
        settings (AppSettings): What the router does for the calling app. It never carries a secret.
    """

    reattach: list[str] | None
    settings: AppSettings
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reattach: list[str] | None
        if isinstance(self.reattach, list):
            reattach = self.reattach

        else:
            reattach = self.reattach

        settings = self.settings.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "reattach": reattach,
                "settings": settings,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.app_settings import AppSettings

        d = dict(src_dict)

        def _parse_reattach(data: object) -> list[str] | None:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                reattach_type_0 = cast(list[str], data)

                return reattach_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None, data)

        reattach = _parse_reattach(d.pop("reattach"))

        settings = AppSettings.from_dict(d.pop("settings"))

        stream_check = cls(
            reattach=reattach,
            settings=settings,
        )

        stream_check.additional_properties = d
        return stream_check

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
