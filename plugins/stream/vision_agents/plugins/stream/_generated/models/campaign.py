from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.campaign_state import CampaignState
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.campaign_tags import CampaignTags


T = TypeVar("T", bound="Campaign")


@_attrs_define
class Campaign:
    """
    Attributes:
        id (str):
        name (str):
        config_id (str):
        from_number (str):
        concurrency (int):
        state (CampaignState):
        created_at (datetime.datetime):
        tags (CampaignTags | Unset):
        started_at (datetime.datetime | Unset):
        finished_at (datetime.datetime | Unset):
    """

    id: str
    name: str
    config_id: str
    from_number: str
    concurrency: int
    state: CampaignState
    created_at: datetime.datetime
    tags: CampaignTags | Unset = UNSET
    started_at: datetime.datetime | Unset = UNSET
    finished_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        config_id = self.config_id

        from_number = self.from_number

        concurrency = self.concurrency

        state = self.state.value

        created_at = self.created_at.isoformat()

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        started_at: str | Unset = UNSET
        if not isinstance(self.started_at, Unset):
            started_at = self.started_at.isoformat()

        finished_at: str | Unset = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "config_id": config_id,
                "from_number": from_number,
                "concurrency": concurrency,
                "state": state,
                "created_at": created_at,
            }
        )
        if tags is not UNSET:
            field_dict["tags"] = tags
        if started_at is not UNSET:
            field_dict["started_at"] = started_at
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.campaign_tags import CampaignTags

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        config_id = d.pop("config_id")

        from_number = d.pop("from_number")

        concurrency = d.pop("concurrency")

        state = CampaignState(d.pop("state"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        _tags = d.pop("tags", UNSET)
        tags: CampaignTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CampaignTags.from_dict(_tags)

        _started_at = d.pop("started_at", UNSET)
        started_at: datetime.datetime | Unset
        if isinstance(_started_at, Unset):
            started_at = UNSET
        else:
            started_at = datetime.datetime.fromisoformat(_started_at)

        _finished_at = d.pop("finished_at", UNSET)
        finished_at: datetime.datetime | Unset
        if isinstance(_finished_at, Unset):
            finished_at = UNSET
        else:
            finished_at = datetime.datetime.fromisoformat(_finished_at)

        campaign = cls(
            id=id,
            name=name,
            config_id=config_id,
            from_number=from_number,
            concurrency=concurrency,
            state=state,
            created_at=created_at,
            tags=tags,
            started_at=started_at,
            finished_at=finished_at,
        )

        campaign.additional_properties = d
        return campaign

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
