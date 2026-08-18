from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.granularity import Granularity
from ..types import UNSET, Unset

T = TypeVar("T", bound="RollupRequest")


@_attrs_define
class RollupRequest:
    """
    Attributes:
        from_ (datetime.datetime):
        to (datetime.datetime):
        granularity (Granularity | Unset):
    """

    from_: datetime.datetime
    to: datetime.datetime
    granularity: Granularity | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from_ = self.from_.isoformat()

        to = self.to.isoformat()

        granularity: str | Unset = UNSET
        if not isinstance(self.granularity, Unset):
            granularity = self.granularity.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "from": from_,
                "to": to,
            }
        )
        if granularity is not UNSET:
            field_dict["granularity"] = granularity

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        from_ = datetime.datetime.fromisoformat(d.pop("from"))

        to = datetime.datetime.fromisoformat(d.pop("to"))

        _granularity = d.pop("granularity", UNSET)
        granularity: Granularity | Unset
        if isinstance(_granularity, Unset):
            granularity = UNSET
        else:
            granularity = Granularity(_granularity)

        rollup_request = cls(
            from_=from_,
            to=to,
            granularity=granularity,
        )

        rollup_request.additional_properties = d
        return rollup_request

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
