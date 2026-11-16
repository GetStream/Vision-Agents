from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="VoiceProfile")


@_attrs_define
class VoiceProfile:
    """
    Attributes:
        call_recording_enabled (bool | Unset):
        caller_id_number (str | Unset): A number bought here, or a verified external one.
        calling_purpose (str | Unset): Support, reminders, sales and so on.
        consent_collection_method (str | Unset):
        consent_disclosure_text (str | Unset):
        consent_evidence_location (str | Unset):
        destination_countries (list[str] | None | Unset):
        expected_call_volume (str | Unset):
        opt_out_handling (str | Unset):
        recording_disclosure (str | Unset): How a recorded call is disclosed and consented to.
    """

    call_recording_enabled: bool | Unset = UNSET
    caller_id_number: str | Unset = UNSET
    calling_purpose: str | Unset = UNSET
    consent_collection_method: str | Unset = UNSET
    consent_disclosure_text: str | Unset = UNSET
    consent_evidence_location: str | Unset = UNSET
    destination_countries: list[str] | None | Unset = UNSET
    expected_call_volume: str | Unset = UNSET
    opt_out_handling: str | Unset = UNSET
    recording_disclosure: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        call_recording_enabled = self.call_recording_enabled

        caller_id_number = self.caller_id_number

        calling_purpose = self.calling_purpose

        consent_collection_method = self.consent_collection_method

        consent_disclosure_text = self.consent_disclosure_text

        consent_evidence_location = self.consent_evidence_location

        destination_countries: list[str] | None | Unset
        if isinstance(self.destination_countries, Unset):
            destination_countries = UNSET
        elif isinstance(self.destination_countries, list):
            destination_countries = self.destination_countries

        else:
            destination_countries = self.destination_countries

        expected_call_volume = self.expected_call_volume

        opt_out_handling = self.opt_out_handling

        recording_disclosure = self.recording_disclosure

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if call_recording_enabled is not UNSET:
            field_dict["call_recording_enabled"] = call_recording_enabled
        if caller_id_number is not UNSET:
            field_dict["caller_id_number"] = caller_id_number
        if calling_purpose is not UNSET:
            field_dict["calling_purpose"] = calling_purpose
        if consent_collection_method is not UNSET:
            field_dict["consent_collection_method"] = consent_collection_method
        if consent_disclosure_text is not UNSET:
            field_dict["consent_disclosure_text"] = consent_disclosure_text
        if consent_evidence_location is not UNSET:
            field_dict["consent_evidence_location"] = consent_evidence_location
        if destination_countries is not UNSET:
            field_dict["destination_countries"] = destination_countries
        if expected_call_volume is not UNSET:
            field_dict["expected_call_volume"] = expected_call_volume
        if opt_out_handling is not UNSET:
            field_dict["opt_out_handling"] = opt_out_handling
        if recording_disclosure is not UNSET:
            field_dict["recording_disclosure"] = recording_disclosure

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        call_recording_enabled = d.pop("call_recording_enabled", UNSET)

        caller_id_number = d.pop("caller_id_number", UNSET)

        calling_purpose = d.pop("calling_purpose", UNSET)

        consent_collection_method = d.pop("consent_collection_method", UNSET)

        consent_disclosure_text = d.pop("consent_disclosure_text", UNSET)

        consent_evidence_location = d.pop("consent_evidence_location", UNSET)

        def _parse_destination_countries(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                destination_countries_type_0 = cast(list[str], data)

                return destination_countries_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        destination_countries = _parse_destination_countries(
            d.pop("destination_countries", UNSET)
        )

        expected_call_volume = d.pop("expected_call_volume", UNSET)

        opt_out_handling = d.pop("opt_out_handling", UNSET)

        recording_disclosure = d.pop("recording_disclosure", UNSET)

        voice_profile = cls(
            call_recording_enabled=call_recording_enabled,
            caller_id_number=caller_id_number,
            calling_purpose=calling_purpose,
            consent_collection_method=consent_collection_method,
            consent_disclosure_text=consent_disclosure_text,
            consent_evidence_location=consent_evidence_location,
            destination_countries=destination_countries,
            expected_call_volume=expected_call_volume,
            opt_out_handling=opt_out_handling,
            recording_disclosure=recording_disclosure,
        )

        voice_profile.additional_properties = d
        return voice_profile

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
