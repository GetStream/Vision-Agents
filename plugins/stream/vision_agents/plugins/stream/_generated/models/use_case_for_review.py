from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.use_case_status import UseCaseStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.business_profile import BusinessProfile
    from ..models.use_case_channels import UseCaseChannels
    from ..models.use_case_review import UseCaseReview


T = TypeVar("T", bound="UseCaseForReview")


@_attrs_define
class UseCaseForReview:
    """A use case with the app it is for and the profile it was submitted on.

    Attributes:
        created_at (datetime.datetime):
        customer_id (str): The app the use case is for.
        id (str):
        name (str):
        status (UseCaseStatus): Where a use case stands. draft, changes_requested and vendor_rejected can be edited and
            submitted; submitted waits on Stream's review; vendor_pending on the vendor's; approved numbers may send.
            rejected is final.
        updated_at (datetime.datetime):
        age_gated (bool | Unset):
        approved_at (datetime.datetime | Unset):
        business_profile (BusinessProfile | Unset): Who the app is, and the brand a vendor registered it as.
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
        reviews (list[UseCaseReview] | None | Unset): What happened to it so far, oldest first. Only on a single use
            case.
        submitted_at (datetime.datetime | Unset):
        use_case_type (str | Unset): The campaign registry's use case, such as CUSTOMER_CARE, ACCOUNT_NOTIFICATION, 2FA,
            MARKETING or MIXED.
        vendor (str | Unset): Who registers the campaign, once Stream approved it.
        vendor_campaign_id (str | Unset):
        vendor_status (str | Unset): The vendor's word for where the campaign stands, such as TCR_ACCEPTED.
    """

    created_at: datetime.datetime
    customer_id: str
    id: str
    name: str
    status: UseCaseStatus
    updated_at: datetime.datetime
    age_gated: bool | Unset = UNSET
    approved_at: datetime.datetime | Unset = UNSET
    business_profile: BusinessProfile | Unset = UNSET
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
    reviews: list[UseCaseReview] | None | Unset = UNSET
    submitted_at: datetime.datetime | Unset = UNSET
    use_case_type: str | Unset = UNSET
    vendor: str | Unset = UNSET
    vendor_campaign_id: str | Unset = UNSET
    vendor_status: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        customer_id = self.customer_id

        id = self.id

        name = self.name

        status = self.status.value

        updated_at = self.updated_at.isoformat()

        age_gated = self.age_gated

        approved_at: str | Unset = UNSET
        if not isinstance(self.approved_at, Unset):
            approved_at = self.approved_at.isoformat()

        business_profile: dict[str, Any] | Unset = UNSET
        if not isinstance(self.business_profile, Unset):
            business_profile = self.business_profile.to_dict()

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

        reviews: list[dict[str, Any]] | None | Unset
        if isinstance(self.reviews, Unset):
            reviews = UNSET
        elif isinstance(self.reviews, list):
            reviews = []
            for reviews_type_0_item_data in self.reviews:
                reviews_type_0_item = reviews_type_0_item_data.to_dict()
                reviews.append(reviews_type_0_item)

        else:
            reviews = self.reviews

        submitted_at: str | Unset = UNSET
        if not isinstance(self.submitted_at, Unset):
            submitted_at = self.submitted_at.isoformat()

        use_case_type = self.use_case_type

        vendor = self.vendor

        vendor_campaign_id = self.vendor_campaign_id

        vendor_status = self.vendor_status

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "customer_id": customer_id,
                "id": id,
                "name": name,
                "status": status,
                "updated_at": updated_at,
            }
        )
        if age_gated is not UNSET:
            field_dict["age_gated"] = age_gated
        if approved_at is not UNSET:
            field_dict["approved_at"] = approved_at
        if business_profile is not UNSET:
            field_dict["business_profile"] = business_profile
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
        if reviews is not UNSET:
            field_dict["reviews"] = reviews
        if submitted_at is not UNSET:
            field_dict["submitted_at"] = submitted_at
        if use_case_type is not UNSET:
            field_dict["use_case_type"] = use_case_type
        if vendor is not UNSET:
            field_dict["vendor"] = vendor
        if vendor_campaign_id is not UNSET:
            field_dict["vendor_campaign_id"] = vendor_campaign_id
        if vendor_status is not UNSET:
            field_dict["vendor_status"] = vendor_status

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.business_profile import BusinessProfile
        from ..models.use_case_channels import UseCaseChannels
        from ..models.use_case_review import UseCaseReview

        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        customer_id = d.pop("customer_id")

        id = d.pop("id")

        name = d.pop("name")

        status = UseCaseStatus(d.pop("status"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        age_gated = d.pop("age_gated", UNSET)

        _approved_at = d.pop("approved_at", UNSET)
        approved_at: datetime.datetime | Unset
        if isinstance(_approved_at, Unset):
            approved_at = UNSET
        else:
            approved_at = datetime.datetime.fromisoformat(_approved_at)

        _business_profile = d.pop("business_profile", UNSET)
        business_profile: BusinessProfile | Unset
        if isinstance(_business_profile, Unset):
            business_profile = UNSET
        else:
            business_profile = BusinessProfile.from_dict(_business_profile)

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

        def _parse_reviews(data: object) -> list[UseCaseReview] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                reviews_type_0 = []
                _reviews_type_0 = data
                for reviews_type_0_item_data in _reviews_type_0:
                    reviews_type_0_item = UseCaseReview.from_dict(
                        reviews_type_0_item_data
                    )

                    reviews_type_0.append(reviews_type_0_item)

                return reviews_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[UseCaseReview] | None | Unset, data)

        reviews = _parse_reviews(d.pop("reviews", UNSET))

        _submitted_at = d.pop("submitted_at", UNSET)
        submitted_at: datetime.datetime | Unset
        if isinstance(_submitted_at, Unset):
            submitted_at = UNSET
        else:
            submitted_at = datetime.datetime.fromisoformat(_submitted_at)

        use_case_type = d.pop("use_case_type", UNSET)

        vendor = d.pop("vendor", UNSET)

        vendor_campaign_id = d.pop("vendor_campaign_id", UNSET)

        vendor_status = d.pop("vendor_status", UNSET)

        use_case_for_review = cls(
            created_at=created_at,
            customer_id=customer_id,
            id=id,
            name=name,
            status=status,
            updated_at=updated_at,
            age_gated=age_gated,
            approved_at=approved_at,
            business_profile=business_profile,
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
            reviews=reviews,
            submitted_at=submitted_at,
            use_case_type=use_case_type,
            vendor=vendor,
            vendor_campaign_id=vendor_campaign_id,
            vendor_status=vendor_status,
        )

        use_case_for_review.additional_properties = d
        return use_case_for_review

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
