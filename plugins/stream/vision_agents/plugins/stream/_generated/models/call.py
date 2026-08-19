from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.call_direction import CallDirection
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.call_tags import CallTags


T = TypeVar("T", bound="Call")


@_attrs_define
class Call:
    """
    Attributes:
        id (str): The session that ran the call, which is what it is held by.
        call_id (str):
        agent_id (str): Which agent ran it, and where its transcript is kept.
        direction (CallDirection):
        started_at (datetime.datetime):
        config_id (str | Unset):
        campaign_id (str | Unset):
        contact_id (str | Unset):
        from_number (str | Unset):
        to_number (str | Unset):
        ended_at (datetime.datetime | Unset): Absent while the call is still running.
        summary (str | Unset): What a model made of the call, written once it was over.
        review_score (int | Unset): How well the agent handled it, from 1 to 5.
        review_notes (str | Unset):
        tags (CallTags | Unset):
    """

    id: str
    call_id: str
    agent_id: str
    direction: CallDirection
    started_at: datetime.datetime
    config_id: str | Unset = UNSET
    campaign_id: str | Unset = UNSET
    contact_id: str | Unset = UNSET
    from_number: str | Unset = UNSET
    to_number: str | Unset = UNSET
    ended_at: datetime.datetime | Unset = UNSET
    summary: str | Unset = UNSET
    review_score: int | Unset = UNSET
    review_notes: str | Unset = UNSET
    tags: CallTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        call_id = self.call_id

        agent_id = self.agent_id

        direction = self.direction.value

        started_at = self.started_at.isoformat()

        config_id = self.config_id

        campaign_id = self.campaign_id

        contact_id = self.contact_id

        from_number = self.from_number

        to_number = self.to_number

        ended_at: str | Unset = UNSET
        if not isinstance(self.ended_at, Unset):
            ended_at = self.ended_at.isoformat()

        summary = self.summary

        review_score = self.review_score

        review_notes = self.review_notes

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "call_id": call_id,
                "agent_id": agent_id,
                "direction": direction,
                "started_at": started_at,
            }
        )
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if campaign_id is not UNSET:
            field_dict["campaign_id"] = campaign_id
        if contact_id is not UNSET:
            field_dict["contact_id"] = contact_id
        if from_number is not UNSET:
            field_dict["from_number"] = from_number
        if to_number is not UNSET:
            field_dict["to_number"] = to_number
        if ended_at is not UNSET:
            field_dict["ended_at"] = ended_at
        if summary is not UNSET:
            field_dict["summary"] = summary
        if review_score is not UNSET:
            field_dict["review_score"] = review_score
        if review_notes is not UNSET:
            field_dict["review_notes"] = review_notes
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.call_tags import CallTags

        d = dict(src_dict)
        id = d.pop("id")

        call_id = d.pop("call_id")

        agent_id = d.pop("agent_id")

        direction = CallDirection(d.pop("direction"))

        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        config_id = d.pop("config_id", UNSET)

        campaign_id = d.pop("campaign_id", UNSET)

        contact_id = d.pop("contact_id", UNSET)

        from_number = d.pop("from_number", UNSET)

        to_number = d.pop("to_number", UNSET)

        _ended_at = d.pop("ended_at", UNSET)
        ended_at: datetime.datetime | Unset
        if isinstance(_ended_at, Unset):
            ended_at = UNSET
        else:
            ended_at = datetime.datetime.fromisoformat(_ended_at)

        summary = d.pop("summary", UNSET)

        review_score = d.pop("review_score", UNSET)

        review_notes = d.pop("review_notes", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: CallTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CallTags.from_dict(_tags)

        call = cls(
            id=id,
            call_id=call_id,
            agent_id=agent_id,
            direction=direction,
            started_at=started_at,
            config_id=config_id,
            campaign_id=campaign_id,
            contact_id=contact_id,
            from_number=from_number,
            to_number=to_number,
            ended_at=ended_at,
            summary=summary,
            review_score=review_score,
            review_notes=review_notes,
            tags=tags,
        )

        call.additional_properties = d
        return call

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
