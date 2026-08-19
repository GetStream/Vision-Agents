from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.campaign_request_tags import CampaignRequestTags


T = TypeVar("T", bound="CampaignRequest")


@_attrs_define
class CampaignRequest:
    """
    Attributes:
        name (str):
        config_id (str): The agent config the calls are made with.
        from_number (str): One of your own numbers, which is what the person sees.
        concurrency (int | Unset): How many of these calls may be happening at once. Default: 1.
        tags (CampaignRequestTags | Unset):
    """

    name: str
    config_id: str
    from_number: str
    concurrency: int | Unset = 1
    tags: CampaignRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        config_id = self.config_id

        from_number = self.from_number

        concurrency = self.concurrency

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "config_id": config_id,
                "from_number": from_number,
            }
        )
        if concurrency is not UNSET:
            field_dict["concurrency"] = concurrency
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.campaign_request_tags import CampaignRequestTags

        d = dict(src_dict)
        name = d.pop("name")

        config_id = d.pop("config_id")

        from_number = d.pop("from_number")

        concurrency = d.pop("concurrency", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: CampaignRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CampaignRequestTags.from_dict(_tags)

        campaign_request = cls(
            name=name,
            config_id=config_id,
            from_number=from_number,
            concurrency=concurrency,
            tags=tags,
        )

        campaign_request.additional_properties = d
        return campaign_request

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
