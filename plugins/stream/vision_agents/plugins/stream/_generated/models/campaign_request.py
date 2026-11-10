from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.campaign_type import CampaignType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.calling_hours import CallingHours
    from ..models.campaign_request_tags import CampaignRequestTags
    from ..models.campaign_sequence import CampaignSequence


T = TypeVar("T", bound="CampaignRequest")


@_attrs_define
class CampaignRequest:
    """
    Attributes:
        calling_hours (CallingHours | Unset):
        concurrency (int | Unset): How many of these calls may be happening at once. Omitted is 1 on create.
        config_id (str | Unset): The agent config the calls are made with. Setting it sets the sequence's step's too.
        default_country (str | Unset): The region, as ISO 3166-1 alpha-2, a contact's number without a country code is
            read in.
        from_number (str | Unset): One of the app's own numbers, which is what the person sees.
        name (str | Unset): Required on create.
        sequence (CampaignSequence | Unset):
        tags (CampaignRequestTags | Unset):
        type_ (CampaignType | Unset): What a campaign rings. standard rings the contacts it was given; auto_enroll is
            refused for now. Fixed once the campaign is created.
    """

    calling_hours: CallingHours | Unset = UNSET
    concurrency: int | Unset = UNSET
    config_id: str | Unset = UNSET
    default_country: str | Unset = UNSET
    from_number: str | Unset = UNSET
    name: str | Unset = UNSET
    sequence: CampaignSequence | Unset = UNSET
    tags: CampaignRequestTags | Unset = UNSET
    type_: CampaignType | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        calling_hours: dict[str, Any] | Unset = UNSET
        if not isinstance(self.calling_hours, Unset):
            calling_hours = self.calling_hours.to_dict()

        concurrency = self.concurrency

        config_id = self.config_id

        default_country = self.default_country

        from_number = self.from_number

        name = self.name

        sequence: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sequence, Unset):
            sequence = self.sequence.to_dict()

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        type_: str | Unset = UNSET
        if not isinstance(self.type_, Unset):
            type_ = self.type_.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if calling_hours is not UNSET:
            field_dict["calling_hours"] = calling_hours
        if concurrency is not UNSET:
            field_dict["concurrency"] = concurrency
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if default_country is not UNSET:
            field_dict["default_country"] = default_country
        if from_number is not UNSET:
            field_dict["from_number"] = from_number
        if name is not UNSET:
            field_dict["name"] = name
        if sequence is not UNSET:
            field_dict["sequence"] = sequence
        if tags is not UNSET:
            field_dict["tags"] = tags
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.calling_hours import CallingHours
        from ..models.campaign_request_tags import CampaignRequestTags
        from ..models.campaign_sequence import CampaignSequence

        d = dict(src_dict)
        _calling_hours = d.pop("calling_hours", UNSET)
        calling_hours: CallingHours | Unset
        if isinstance(_calling_hours, Unset):
            calling_hours = UNSET
        else:
            calling_hours = CallingHours.from_dict(_calling_hours)

        concurrency = d.pop("concurrency", UNSET)

        config_id = d.pop("config_id", UNSET)

        default_country = d.pop("default_country", UNSET)

        from_number = d.pop("from_number", UNSET)

        name = d.pop("name", UNSET)

        _sequence = d.pop("sequence", UNSET)
        sequence: CampaignSequence | Unset
        if isinstance(_sequence, Unset):
            sequence = UNSET
        else:
            sequence = CampaignSequence.from_dict(_sequence)

        _tags = d.pop("tags", UNSET)
        tags: CampaignRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CampaignRequestTags.from_dict(_tags)

        _type_ = d.pop("type", UNSET)
        type_: CampaignType | Unset
        if isinstance(_type_, Unset):
            type_ = UNSET
        else:
            type_ = CampaignType._decode(_type_)

        campaign_request = cls(
            calling_hours=calling_hours,
            concurrency=concurrency,
            config_id=config_id,
            default_country=default_country,
            from_number=from_number,
            name=name,
            sequence=sequence,
            tags=tags,
            type_=type_,
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
