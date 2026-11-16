from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.campaign_attempt import CampaignAttempt
    from ..models.campaign_outcome import CampaignOutcome
    from ..models.campaign_route import CampaignRoute
    from ..models.campaign_voicemail import CampaignVoicemail


T = TypeVar("T", bound="CampaignStep")


@_attrs_define
class CampaignStep:
    """
    Attributes:
        attempts (list[CampaignAttempt]): 1 to 10. The first has a delay of 0.
        config_id (str): The agent config the step rings with, the same as the campaign's config_id.
        id (str): step_1, for now.
        no_response (CampaignRoute):
        other (CampaignRoute):
        outcomes (list[CampaignOutcome]): What an answered call can end as. do_not_call is built in.
        goal (str | Unset): What this conversation is for, told to the agent on every call of the step. {{variable}}
            placeholders are filled from the contact. Required to launch.
        voicemail (CampaignVoicemail | Unset):
    """

    attempts: list[CampaignAttempt]
    config_id: str
    id: str
    no_response: CampaignRoute
    other: CampaignRoute
    outcomes: list[CampaignOutcome]
    goal: str | Unset = UNSET
    voicemail: CampaignVoicemail | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        attempts = []
        for attempts_item_data in self.attempts:
            attempts_item = attempts_item_data.to_dict()
            attempts.append(attempts_item)

        config_id = self.config_id

        id = self.id

        no_response = self.no_response.to_dict()

        other = self.other.to_dict()

        outcomes = []
        for outcomes_item_data in self.outcomes:
            outcomes_item = outcomes_item_data.to_dict()
            outcomes.append(outcomes_item)

        goal = self.goal

        voicemail: dict[str, Any] | Unset = UNSET
        if not isinstance(self.voicemail, Unset):
            voicemail = self.voicemail.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "attempts": attempts,
                "config_id": config_id,
                "id": id,
                "no_response": no_response,
                "other": other,
                "outcomes": outcomes,
            }
        )
        if goal is not UNSET:
            field_dict["goal"] = goal
        if voicemail is not UNSET:
            field_dict["voicemail"] = voicemail

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.campaign_attempt import CampaignAttempt
        from ..models.campaign_outcome import CampaignOutcome
        from ..models.campaign_route import CampaignRoute
        from ..models.campaign_voicemail import CampaignVoicemail

        d = dict(src_dict)
        attempts = []
        _attempts = d.pop("attempts")
        for attempts_item_data in _attempts:
            attempts_item = CampaignAttempt.from_dict(attempts_item_data)

            attempts.append(attempts_item)

        config_id = d.pop("config_id")

        id = d.pop("id")

        no_response = CampaignRoute.from_dict(d.pop("no_response"))

        other = CampaignRoute.from_dict(d.pop("other"))

        outcomes = []
        _outcomes = d.pop("outcomes")
        for outcomes_item_data in _outcomes:
            outcomes_item = CampaignOutcome.from_dict(outcomes_item_data)

            outcomes.append(outcomes_item)

        goal = d.pop("goal", UNSET)

        _voicemail = d.pop("voicemail", UNSET)
        voicemail: CampaignVoicemail | Unset
        if isinstance(_voicemail, Unset):
            voicemail = UNSET
        else:
            voicemail = CampaignVoicemail.from_dict(_voicemail)

        campaign_step = cls(
            attempts=attempts,
            config_id=config_id,
            id=id,
            no_response=no_response,
            other=other,
            outcomes=outcomes,
            goal=goal,
            voicemail=voicemail,
        )

        campaign_step.additional_properties = d
        return campaign_step

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
