from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.use_case_channels import UseCaseChannels


T = TypeVar("T", bound="UseCaseRequest")


@_attrs_define
class UseCaseRequest:
    """What an app sends texts and makes calls for. Saved as a draft and submitted for review once it and the business
    profile are complete.

        Attributes:
            name (str):
            age_gated (bool | Unset):
            channels (UseCaseChannels | Unset):
            description (str | Unset): What the messages are for, in at least 40 characters.
            direct_lending (bool | Unset):
            embedded_links (bool | Unset):
            embedded_phone (bool | Unset):
            help_message (str | Unset): The answer to HELP: who you are and how to reach support.
            is_default (bool | Unset): Send as this use case from every number assigned to no other one. An app's first use
                case is its default.
            message_flow (str | Unset): How a recipient opts in, and where a reviewer can see it, in at least 40 characters.
            message_samples (list[str] | None | Unset): Two to five messages as they will be sent.
            numbers (list[str] | None | Unset): The app's numbers that send as this use case, in E.164. Omitted on an update
                leaves them as they are; empty unassigns them all.
            opt_in_message (str | Unset): The answer to START, and the first message after opting in.
            opt_out_message (str | Unset): The answer to STOP.
            use_case_type (str | Unset): The campaign registry's use case, such as CUSTOMER_CARE, ACCOUNT_NOTIFICATION, 2FA,
                MARKETING or MIXED.
    """

    name: str
    age_gated: bool | Unset = UNSET
    channels: UseCaseChannels | Unset = UNSET
    description: str | Unset = UNSET
    direct_lending: bool | Unset = UNSET
    embedded_links: bool | Unset = UNSET
    embedded_phone: bool | Unset = UNSET
    help_message: str | Unset = UNSET
    is_default: bool | Unset = UNSET
    message_flow: str | Unset = UNSET
    message_samples: list[str] | None | Unset = UNSET
    numbers: list[str] | None | Unset = UNSET
    opt_in_message: str | Unset = UNSET
    opt_out_message: str | Unset = UNSET
    use_case_type: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        age_gated = self.age_gated

        channels: dict[str, Any] | Unset = UNSET
        if not isinstance(self.channels, Unset):
            channels = self.channels.to_dict()

        description = self.description

        direct_lending = self.direct_lending

        embedded_links = self.embedded_links

        embedded_phone = self.embedded_phone

        help_message = self.help_message

        is_default = self.is_default

        message_flow = self.message_flow

        message_samples: list[str] | None | Unset
        if isinstance(self.message_samples, Unset):
            message_samples = UNSET
        elif isinstance(self.message_samples, list):
            message_samples = self.message_samples

        else:
            message_samples = self.message_samples

        numbers: list[str] | None | Unset
        if isinstance(self.numbers, Unset):
            numbers = UNSET
        elif isinstance(self.numbers, list):
            numbers = self.numbers

        else:
            numbers = self.numbers

        opt_in_message = self.opt_in_message

        opt_out_message = self.opt_out_message

        use_case_type = self.use_case_type

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
            }
        )
        if age_gated is not UNSET:
            field_dict["age_gated"] = age_gated
        if channels is not UNSET:
            field_dict["channels"] = channels
        if description is not UNSET:
            field_dict["description"] = description
        if direct_lending is not UNSET:
            field_dict["direct_lending"] = direct_lending
        if embedded_links is not UNSET:
            field_dict["embedded_links"] = embedded_links
        if embedded_phone is not UNSET:
            field_dict["embedded_phone"] = embedded_phone
        if help_message is not UNSET:
            field_dict["help_message"] = help_message
        if is_default is not UNSET:
            field_dict["is_default"] = is_default
        if message_flow is not UNSET:
            field_dict["message_flow"] = message_flow
        if message_samples is not UNSET:
            field_dict["message_samples"] = message_samples
        if numbers is not UNSET:
            field_dict["numbers"] = numbers
        if opt_in_message is not UNSET:
            field_dict["opt_in_message"] = opt_in_message
        if opt_out_message is not UNSET:
            field_dict["opt_out_message"] = opt_out_message
        if use_case_type is not UNSET:
            field_dict["use_case_type"] = use_case_type

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.use_case_channels import UseCaseChannels

        d = dict(src_dict)
        name = d.pop("name")

        age_gated = d.pop("age_gated", UNSET)

        _channels = d.pop("channels", UNSET)
        channels: UseCaseChannels | Unset
        if isinstance(_channels, Unset):
            channels = UNSET
        else:
            channels = UseCaseChannels.from_dict(_channels)

        description = d.pop("description", UNSET)

        direct_lending = d.pop("direct_lending", UNSET)

        embedded_links = d.pop("embedded_links", UNSET)

        embedded_phone = d.pop("embedded_phone", UNSET)

        help_message = d.pop("help_message", UNSET)

        is_default = d.pop("is_default", UNSET)

        message_flow = d.pop("message_flow", UNSET)

        def _parse_message_samples(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                message_samples_type_0 = cast(list[str], data)

                return message_samples_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        message_samples = _parse_message_samples(d.pop("message_samples", UNSET))

        def _parse_numbers(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                numbers_type_0 = cast(list[str], data)

                return numbers_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        numbers = _parse_numbers(d.pop("numbers", UNSET))

        opt_in_message = d.pop("opt_in_message", UNSET)

        opt_out_message = d.pop("opt_out_message", UNSET)

        use_case_type = d.pop("use_case_type", UNSET)

        use_case_request = cls(
            name=name,
            age_gated=age_gated,
            channels=channels,
            description=description,
            direct_lending=direct_lending,
            embedded_links=embedded_links,
            embedded_phone=embedded_phone,
            help_message=help_message,
            is_default=is_default,
            message_flow=message_flow,
            message_samples=message_samples,
            numbers=numbers,
            opt_in_message=opt_in_message,
            opt_out_message=opt_out_message,
            use_case_type=use_case_type,
        )

        use_case_request.additional_properties = d
        return use_case_request

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
