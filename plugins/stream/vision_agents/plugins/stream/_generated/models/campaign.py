from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.campaign_state import CampaignState
from ..models.campaign_type import CampaignType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.calling_hours import CallingHours
    from ..models.campaign_sequence import CampaignSequence
    from ..models.campaign_tags import CampaignTags
    from ..models.contact_counts import ContactCounts


T = TypeVar("T", bound="Campaign")


@_attrs_define
class Campaign:
    """
    Attributes:
        calling_hours (CallingHours):
        concurrency (int): How many of these calls may be happening at once.
        config_id (str): The agent config the calls are made with, the same as the sequence's step names.
        contacts (ContactCounts):
        created_at (datetime.datetime):
        default_country (str): The region, as ISO 3166-1 alpha-2, a contact's number without a country code is read in.
        from_number (str): One of the app's own numbers, which is what the person sees.
        id (str):
        name (str):
        sequence (CampaignSequence):
        state (CampaignState): Where a campaign is: draft until it is launched, scheduled until its start_at, running
            while it rings people, paused while it rings nobody new, stopped for good, completed once it has nobody left to
            ring.
        tags (CampaignTags):
        type_ (CampaignType): What a campaign rings. standard rings the contacts it was given; auto_enroll is refused
            for now. Fixed once the campaign is created.
        completed_at (datetime.datetime | Unset): When it had nobody left to ring.
        launched_at (datetime.datetime | Unset):
        paused_reason (str | Unset): Why the router paused the campaign itself: the code of what refused its next call,
            for example telephony_suspended, account_suspended, insufficient_credits, system_limit_reached,
            spend_limit_reached, user_limit_reached or spend_check_unavailable, config_missing when its agent config was
            deleted, session_refused when its agent config makes a session that cannot open, or voicemail_needs_exact_words
            when its step leaves a voicemail message and its agent config became speech-to-speech. The contact that call was
            for is rung once the campaign is resumed. Absent for a campaign somebody paused, and cleared by resuming it.
        start_at (datetime.datetime | Unset): When it was launched to start ringing.
        started_at (datetime.datetime | Unset): When it first rang anybody.
        stopped_at (datetime.datetime | Unset):
    """

    calling_hours: CallingHours
    concurrency: int
    config_id: str
    contacts: ContactCounts
    created_at: datetime.datetime
    default_country: str
    from_number: str
    id: str
    name: str
    sequence: CampaignSequence
    state: CampaignState
    tags: CampaignTags
    type_: CampaignType
    completed_at: datetime.datetime | Unset = UNSET
    launched_at: datetime.datetime | Unset = UNSET
    paused_reason: str | Unset = UNSET
    start_at: datetime.datetime | Unset = UNSET
    started_at: datetime.datetime | Unset = UNSET
    stopped_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        calling_hours = self.calling_hours.to_dict()

        concurrency = self.concurrency

        config_id = self.config_id

        contacts = self.contacts.to_dict()

        created_at = self.created_at.isoformat()

        default_country = self.default_country

        from_number = self.from_number

        id = self.id

        name = self.name

        sequence = self.sequence.to_dict()

        state = self.state.value

        tags = self.tags.to_dict()

        type_ = self.type_.value

        completed_at: str | Unset = UNSET
        if not isinstance(self.completed_at, Unset):
            completed_at = self.completed_at.isoformat()

        launched_at: str | Unset = UNSET
        if not isinstance(self.launched_at, Unset):
            launched_at = self.launched_at.isoformat()

        paused_reason = self.paused_reason

        start_at: str | Unset = UNSET
        if not isinstance(self.start_at, Unset):
            start_at = self.start_at.isoformat()

        started_at: str | Unset = UNSET
        if not isinstance(self.started_at, Unset):
            started_at = self.started_at.isoformat()

        stopped_at: str | Unset = UNSET
        if not isinstance(self.stopped_at, Unset):
            stopped_at = self.stopped_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "calling_hours": calling_hours,
                "concurrency": concurrency,
                "config_id": config_id,
                "contacts": contacts,
                "created_at": created_at,
                "default_country": default_country,
                "from_number": from_number,
                "id": id,
                "name": name,
                "sequence": sequence,
                "state": state,
                "tags": tags,
                "type": type_,
            }
        )
        if completed_at is not UNSET:
            field_dict["completed_at"] = completed_at
        if launched_at is not UNSET:
            field_dict["launched_at"] = launched_at
        if paused_reason is not UNSET:
            field_dict["paused_reason"] = paused_reason
        if start_at is not UNSET:
            field_dict["start_at"] = start_at
        if started_at is not UNSET:
            field_dict["started_at"] = started_at
        if stopped_at is not UNSET:
            field_dict["stopped_at"] = stopped_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.calling_hours import CallingHours
        from ..models.campaign_sequence import CampaignSequence
        from ..models.campaign_tags import CampaignTags
        from ..models.contact_counts import ContactCounts

        d = dict(src_dict)
        calling_hours = CallingHours.from_dict(d.pop("calling_hours"))

        concurrency = d.pop("concurrency")

        config_id = d.pop("config_id")

        contacts = ContactCounts.from_dict(d.pop("contacts"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        default_country = d.pop("default_country")

        from_number = d.pop("from_number")

        id = d.pop("id")

        name = d.pop("name")

        sequence = CampaignSequence.from_dict(d.pop("sequence"))

        state = CampaignState(d.pop("state"))

        tags = CampaignTags.from_dict(d.pop("tags"))

        type_ = CampaignType(d.pop("type"))

        _completed_at = d.pop("completed_at", UNSET)
        completed_at: datetime.datetime | Unset
        if isinstance(_completed_at, Unset):
            completed_at = UNSET
        else:
            completed_at = datetime.datetime.fromisoformat(_completed_at)

        _launched_at = d.pop("launched_at", UNSET)
        launched_at: datetime.datetime | Unset
        if isinstance(_launched_at, Unset):
            launched_at = UNSET
        else:
            launched_at = datetime.datetime.fromisoformat(_launched_at)

        paused_reason = d.pop("paused_reason", UNSET)

        _start_at = d.pop("start_at", UNSET)
        start_at: datetime.datetime | Unset
        if isinstance(_start_at, Unset):
            start_at = UNSET
        else:
            start_at = datetime.datetime.fromisoformat(_start_at)

        _started_at = d.pop("started_at", UNSET)
        started_at: datetime.datetime | Unset
        if isinstance(_started_at, Unset):
            started_at = UNSET
        else:
            started_at = datetime.datetime.fromisoformat(_started_at)

        _stopped_at = d.pop("stopped_at", UNSET)
        stopped_at: datetime.datetime | Unset
        if isinstance(_stopped_at, Unset):
            stopped_at = UNSET
        else:
            stopped_at = datetime.datetime.fromisoformat(_stopped_at)

        campaign = cls(
            calling_hours=calling_hours,
            concurrency=concurrency,
            config_id=config_id,
            contacts=contacts,
            created_at=created_at,
            default_country=default_country,
            from_number=from_number,
            id=id,
            name=name,
            sequence=sequence,
            state=state,
            tags=tags,
            type_=type_,
            completed_at=completed_at,
            launched_at=launched_at,
            paused_reason=paused_reason,
            start_at=start_at,
            started_at=started_at,
            stopped_at=stopped_at,
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
