from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.price_item import PriceItem
    from ..models.price_preview_item import PricePreviewItem


T = TypeVar("T", bound="PricePreview")


@_attrs_define
class PricePreview:
    """
    Attributes:
        confirmation_required (bool): createPriceChange answers 409 confirmation_required without confirm: true.
        effective_at (datetime.datetime): When the change would take effect.
        items (list[PricePreviewItem]):
        markup_bps_after (int):
        markup_bps_before (int):
        needs_confirmation (list[PriceItem] | None | Unset): The items that make it so.
    """

    confirmation_required: bool
    effective_at: datetime.datetime
    items: list[PricePreviewItem]
    markup_bps_after: int
    markup_bps_before: int
    needs_confirmation: list[PriceItem] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        confirmation_required = self.confirmation_required

        effective_at = self.effective_at.isoformat()

        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        markup_bps_after = self.markup_bps_after

        markup_bps_before = self.markup_bps_before

        needs_confirmation: list[dict[str, Any]] | None | Unset
        if isinstance(self.needs_confirmation, Unset):
            needs_confirmation = UNSET
        elif isinstance(self.needs_confirmation, list):
            needs_confirmation = []
            for needs_confirmation_type_0_item_data in self.needs_confirmation:
                needs_confirmation_type_0_item = (
                    needs_confirmation_type_0_item_data.to_dict()
                )
                needs_confirmation.append(needs_confirmation_type_0_item)

        else:
            needs_confirmation = self.needs_confirmation

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "confirmation_required": confirmation_required,
                "effective_at": effective_at,
                "items": items,
                "markup_bps_after": markup_bps_after,
                "markup_bps_before": markup_bps_before,
            }
        )
        if needs_confirmation is not UNSET:
            field_dict["needs_confirmation"] = needs_confirmation

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.price_item import PriceItem
        from ..models.price_preview_item import PricePreviewItem

        d = dict(src_dict)
        confirmation_required = d.pop("confirmation_required")

        effective_at = datetime.datetime.fromisoformat(d.pop("effective_at"))

        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = PricePreviewItem.from_dict(items_item_data)

            items.append(items_item)

        markup_bps_after = d.pop("markup_bps_after")

        markup_bps_before = d.pop("markup_bps_before")

        def _parse_needs_confirmation(data: object) -> list[PriceItem] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                needs_confirmation_type_0 = []
                _needs_confirmation_type_0 = data
                for needs_confirmation_type_0_item_data in _needs_confirmation_type_0:
                    needs_confirmation_type_0_item = PriceItem.from_dict(
                        needs_confirmation_type_0_item_data
                    )

                    needs_confirmation_type_0.append(needs_confirmation_type_0_item)

                return needs_confirmation_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[PriceItem] | None | Unset, data)

        needs_confirmation = _parse_needs_confirmation(
            d.pop("needs_confirmation", UNSET)
        )

        price_preview = cls(
            confirmation_required=confirmation_required,
            effective_at=effective_at,
            items=items,
            markup_bps_after=markup_bps_after,
            markup_bps_before=markup_bps_before,
            needs_confirmation=needs_confirmation,
        )

        price_preview.additional_properties = d
        return price_preview

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
