from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="RCSProfile")


@_attrs_define
class RCSProfile:
    """
    Attributes:
        agent_overview (str | Unset):
        brand_color (str | Unset): Hex, such as #1A73E8, with at least 4.5:1 contrast against white.
        call_to_action_text (str | Unset):
        call_to_action_url (str | Unset):
        company_overview (str | Unset):
        description (str | Unset):
        display_name (str | Unset):
        double_opt_in (bool | Unset):
        hero_url (str | Unset): 1440 by 448 PNG or JPEG, at most 200 KB, on public HTTPS.
        interaction_types (str | Unset):
        logo_url (str | Unset): 224 by 224 PNG or JPEG, at most 50 KB, on public HTTPS.
        message_examples (list[str] | None | Unset):
        opt_in_confirmation_message (str | Unset):
        opt_in_methods (str | Unset):
        support_email (str | Unset):
        support_label (str | Unset):
        support_phone (str | Unset):
        test_video_url (str | Unset): Shows consent, example interactions, HELP and STOP.
        use_case (str | Unset): otp, transactional, promotional or multi_use.
    """

    agent_overview: str | Unset = UNSET
    brand_color: str | Unset = UNSET
    call_to_action_text: str | Unset = UNSET
    call_to_action_url: str | Unset = UNSET
    company_overview: str | Unset = UNSET
    description: str | Unset = UNSET
    display_name: str | Unset = UNSET
    double_opt_in: bool | Unset = UNSET
    hero_url: str | Unset = UNSET
    interaction_types: str | Unset = UNSET
    logo_url: str | Unset = UNSET
    message_examples: list[str] | None | Unset = UNSET
    opt_in_confirmation_message: str | Unset = UNSET
    opt_in_methods: str | Unset = UNSET
    support_email: str | Unset = UNSET
    support_label: str | Unset = UNSET
    support_phone: str | Unset = UNSET
    test_video_url: str | Unset = UNSET
    use_case: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent_overview = self.agent_overview

        brand_color = self.brand_color

        call_to_action_text = self.call_to_action_text

        call_to_action_url = self.call_to_action_url

        company_overview = self.company_overview

        description = self.description

        display_name = self.display_name

        double_opt_in = self.double_opt_in

        hero_url = self.hero_url

        interaction_types = self.interaction_types

        logo_url = self.logo_url

        message_examples: list[str] | None | Unset
        if isinstance(self.message_examples, Unset):
            message_examples = UNSET
        elif isinstance(self.message_examples, list):
            message_examples = self.message_examples

        else:
            message_examples = self.message_examples

        opt_in_confirmation_message = self.opt_in_confirmation_message

        opt_in_methods = self.opt_in_methods

        support_email = self.support_email

        support_label = self.support_label

        support_phone = self.support_phone

        test_video_url = self.test_video_url

        use_case = self.use_case

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if agent_overview is not UNSET:
            field_dict["agent_overview"] = agent_overview
        if brand_color is not UNSET:
            field_dict["brand_color"] = brand_color
        if call_to_action_text is not UNSET:
            field_dict["call_to_action_text"] = call_to_action_text
        if call_to_action_url is not UNSET:
            field_dict["call_to_action_url"] = call_to_action_url
        if company_overview is not UNSET:
            field_dict["company_overview"] = company_overview
        if description is not UNSET:
            field_dict["description"] = description
        if display_name is not UNSET:
            field_dict["display_name"] = display_name
        if double_opt_in is not UNSET:
            field_dict["double_opt_in"] = double_opt_in
        if hero_url is not UNSET:
            field_dict["hero_url"] = hero_url
        if interaction_types is not UNSET:
            field_dict["interaction_types"] = interaction_types
        if logo_url is not UNSET:
            field_dict["logo_url"] = logo_url
        if message_examples is not UNSET:
            field_dict["message_examples"] = message_examples
        if opt_in_confirmation_message is not UNSET:
            field_dict["opt_in_confirmation_message"] = opt_in_confirmation_message
        if opt_in_methods is not UNSET:
            field_dict["opt_in_methods"] = opt_in_methods
        if support_email is not UNSET:
            field_dict["support_email"] = support_email
        if support_label is not UNSET:
            field_dict["support_label"] = support_label
        if support_phone is not UNSET:
            field_dict["support_phone"] = support_phone
        if test_video_url is not UNSET:
            field_dict["test_video_url"] = test_video_url
        if use_case is not UNSET:
            field_dict["use_case"] = use_case

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        agent_overview = d.pop("agent_overview", UNSET)

        brand_color = d.pop("brand_color", UNSET)

        call_to_action_text = d.pop("call_to_action_text", UNSET)

        call_to_action_url = d.pop("call_to_action_url", UNSET)

        company_overview = d.pop("company_overview", UNSET)

        description = d.pop("description", UNSET)

        display_name = d.pop("display_name", UNSET)

        double_opt_in = d.pop("double_opt_in", UNSET)

        hero_url = d.pop("hero_url", UNSET)

        interaction_types = d.pop("interaction_types", UNSET)

        logo_url = d.pop("logo_url", UNSET)

        def _parse_message_examples(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                message_examples_type_0 = cast(list[str], data)

                return message_examples_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        message_examples = _parse_message_examples(d.pop("message_examples", UNSET))

        opt_in_confirmation_message = d.pop("opt_in_confirmation_message", UNSET)

        opt_in_methods = d.pop("opt_in_methods", UNSET)

        support_email = d.pop("support_email", UNSET)

        support_label = d.pop("support_label", UNSET)

        support_phone = d.pop("support_phone", UNSET)

        test_video_url = d.pop("test_video_url", UNSET)

        use_case = d.pop("use_case", UNSET)

        rcs_profile = cls(
            agent_overview=agent_overview,
            brand_color=brand_color,
            call_to_action_text=call_to_action_text,
            call_to_action_url=call_to_action_url,
            company_overview=company_overview,
            description=description,
            display_name=display_name,
            double_opt_in=double_opt_in,
            hero_url=hero_url,
            interaction_types=interaction_types,
            logo_url=logo_url,
            message_examples=message_examples,
            opt_in_confirmation_message=opt_in_confirmation_message,
            opt_in_methods=opt_in_methods,
            support_email=support_email,
            support_label=support_label,
            support_phone=support_phone,
            test_video_url=test_video_url,
            use_case=use_case,
        )

        rcs_profile.additional_properties = d
        return rcs_profile

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
