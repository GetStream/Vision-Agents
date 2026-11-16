from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.contact_stage import ContactStage
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.contact_attempt import ContactAttempt
    from ..models.contact_variables import ContactVariables


T = TypeVar("T", bound="Contact")


@_attrs_define
class Contact:
    """
    Attributes:
        attempts_made (int): Attempts that placed at least one call.
        attempts_total (int): The attempts the sequence makes at most.
        converted (bool): They ended successfully.
        created_at (datetime.datetime):
        id (str):
        stage (ContactStage): Where a contact is: enrolled while waiting to be rung, in_progress while a call to them
            happens or is being read, completed once the sequence is done with them, rejected when their number was refused
            before any call, stopped when they were left unrung, failed when a call could not be made.
        to_number (str): E.164.
        variables (ContactVariables): Filled into the agent's {{placeholders}} on their calls.
        zone (str): The IANA zone the calling hours are read in for them: the campaign's fixed zone, else their
            time_zone, else their number's, else the campaign's fallback.
        attempts (list[ContactAttempt] | None | Unset): Every dial made to them, oldest first. Only on a contact read
            alone.
        call_id (str | Unset): The last call placed to them, which is what the call paths take.
        next_attempt_at (datetime.datetime | Unset): When they are next rung, while they are waiting to be.
        outcome (str | Unset): The id of the outcome they ended with, once they have.
        reason (str | Unset): Why they were rejected, failed or stopped.
        time_zone (str | Unset): The IANA zone they were added with.
    """

    attempts_made: int
    attempts_total: int
    converted: bool
    created_at: datetime.datetime
    id: str
    stage: ContactStage
    to_number: str
    variables: ContactVariables
    zone: str
    attempts: list[ContactAttempt] | None | Unset = UNSET
    call_id: str | Unset = UNSET
    next_attempt_at: datetime.datetime | Unset = UNSET
    outcome: str | Unset = UNSET
    reason: str | Unset = UNSET
    time_zone: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        attempts_made = self.attempts_made

        attempts_total = self.attempts_total

        converted = self.converted

        created_at = self.created_at.isoformat()

        id = self.id

        stage = self.stage.value

        to_number = self.to_number

        variables = self.variables.to_dict()

        zone = self.zone

        attempts: list[dict[str, Any]] | None | Unset
        if isinstance(self.attempts, Unset):
            attempts = UNSET
        elif isinstance(self.attempts, list):
            attempts = []
            for attempts_type_0_item_data in self.attempts:
                attempts_type_0_item = attempts_type_0_item_data.to_dict()
                attempts.append(attempts_type_0_item)

        else:
            attempts = self.attempts

        call_id = self.call_id

        next_attempt_at: str | Unset = UNSET
        if not isinstance(self.next_attempt_at, Unset):
            next_attempt_at = self.next_attempt_at.isoformat()

        outcome = self.outcome

        reason = self.reason

        time_zone = self.time_zone

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "attempts_made": attempts_made,
                "attempts_total": attempts_total,
                "converted": converted,
                "created_at": created_at,
                "id": id,
                "stage": stage,
                "to_number": to_number,
                "variables": variables,
                "zone": zone,
            }
        )
        if attempts is not UNSET:
            field_dict["attempts"] = attempts
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if next_attempt_at is not UNSET:
            field_dict["next_attempt_at"] = next_attempt_at
        if outcome is not UNSET:
            field_dict["outcome"] = outcome
        if reason is not UNSET:
            field_dict["reason"] = reason
        if time_zone is not UNSET:
            field_dict["time_zone"] = time_zone

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.contact_attempt import ContactAttempt
        from ..models.contact_variables import ContactVariables

        d = dict(src_dict)
        attempts_made = d.pop("attempts_made")

        attempts_total = d.pop("attempts_total")

        converted = d.pop("converted")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        stage = ContactStage._decode(d.pop("stage"))

        to_number = d.pop("to_number")

        variables = ContactVariables.from_dict(d.pop("variables"))

        zone = d.pop("zone")

        def _parse_attempts(data: object) -> list[ContactAttempt] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                attempts_type_0 = []
                _attempts_type_0 = data
                for attempts_type_0_item_data in _attempts_type_0:
                    attempts_type_0_item = ContactAttempt.from_dict(
                        attempts_type_0_item_data
                    )

                    attempts_type_0.append(attempts_type_0_item)

                return attempts_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ContactAttempt] | None | Unset, data)

        attempts = _parse_attempts(d.pop("attempts", UNSET))

        call_id = d.pop("call_id", UNSET)

        _next_attempt_at = d.pop("next_attempt_at", UNSET)
        next_attempt_at: datetime.datetime | Unset
        if isinstance(_next_attempt_at, Unset):
            next_attempt_at = UNSET
        else:
            next_attempt_at = datetime.datetime.fromisoformat(_next_attempt_at)

        outcome = d.pop("outcome", UNSET)

        reason = d.pop("reason", UNSET)

        time_zone = d.pop("time_zone", UNSET)

        contact = cls(
            attempts_made=attempts_made,
            attempts_total=attempts_total,
            converted=converted,
            created_at=created_at,
            id=id,
            stage=stage,
            to_number=to_number,
            variables=variables,
            zone=zone,
            attempts=attempts,
            call_id=call_id,
            next_attempt_at=next_attempt_at,
            outcome=outcome,
            reason=reason,
            time_zone=time_zone,
        )

        contact.additional_properties = d
        return contact

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
