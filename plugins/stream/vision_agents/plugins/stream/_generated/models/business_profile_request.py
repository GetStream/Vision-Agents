from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.business_profile_request_legal_entity_type import (
    BusinessProfileRequestLegalEntityType,
)
from ..models.business_profile_request_organization_type import (
    BusinessProfileRequestOrganizationType,
)
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.postal_address import PostalAddress


T = TypeVar("T", bound="BusinessProfileRequest")


@_attrs_define
class BusinessProfileRequest:
    """Who the app is: written once and reused by every channel it registers for. Nothing is required to save it;
    submitting a use case says what is missing.

        Attributes:
            authorized_contact_email (str | Unset):
            authorized_contact_first_name (str | Unset):
            authorized_contact_last_name (str | Unset):
            authorized_contact_phone (str | Unset): E.164.
            authorized_contact_title (str | Unset):
            brand_name (str | Unset): What people know it as, shown to recipients. Omitted is the legal name.
            business_registration_country (str | Unset): ISO 3166-1 alpha-2.
            business_verification_documents (list[str] | None | Unset): URLs of documents a reviewer may ask for, such as
                articles of incorporation.
            industry (str | Unset): The registry's vertical, such as technology, healthcare or retail.
            legal_business_name (str | Unset): The name the business is registered under.
            legal_entity_type (BusinessProfileRequestLegalEntityType | Unset):
            organization_type (BusinessProfileRequestOrganizationType | Unset):
            privacy_policy_url (str | Unset):
            registered_address (PostalAddress | Unset):
            stock_exchange (str | Unset): Where it is listed, such as NASDAQ or NYSE.
            stock_symbol (str | Unset): A public company's ticker.
            tax_id (str | Unset): The EIN in the US, or the country's business number. A sole proprietor has none.
            tax_id_issuing_country (str | Unset): ISO 3166-1 alpha-2. Omitted is the registration country.
            terms_and_conditions_url (str | Unset):
            website_url (str | Unset):
    """

    authorized_contact_email: str | Unset = UNSET
    authorized_contact_first_name: str | Unset = UNSET
    authorized_contact_last_name: str | Unset = UNSET
    authorized_contact_phone: str | Unset = UNSET
    authorized_contact_title: str | Unset = UNSET
    brand_name: str | Unset = UNSET
    business_registration_country: str | Unset = UNSET
    business_verification_documents: list[str] | None | Unset = UNSET
    industry: str | Unset = UNSET
    legal_business_name: str | Unset = UNSET
    legal_entity_type: BusinessProfileRequestLegalEntityType | Unset = UNSET
    organization_type: BusinessProfileRequestOrganizationType | Unset = UNSET
    privacy_policy_url: str | Unset = UNSET
    registered_address: PostalAddress | Unset = UNSET
    stock_exchange: str | Unset = UNSET
    stock_symbol: str | Unset = UNSET
    tax_id: str | Unset = UNSET
    tax_id_issuing_country: str | Unset = UNSET
    terms_and_conditions_url: str | Unset = UNSET
    website_url: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        authorized_contact_email = self.authorized_contact_email

        authorized_contact_first_name = self.authorized_contact_first_name

        authorized_contact_last_name = self.authorized_contact_last_name

        authorized_contact_phone = self.authorized_contact_phone

        authorized_contact_title = self.authorized_contact_title

        brand_name = self.brand_name

        business_registration_country = self.business_registration_country

        business_verification_documents: list[str] | None | Unset
        if isinstance(self.business_verification_documents, Unset):
            business_verification_documents = UNSET
        elif isinstance(self.business_verification_documents, list):
            business_verification_documents = self.business_verification_documents

        else:
            business_verification_documents = self.business_verification_documents

        industry = self.industry

        legal_business_name = self.legal_business_name

        legal_entity_type: str | Unset = UNSET
        if not isinstance(self.legal_entity_type, Unset):
            legal_entity_type = self.legal_entity_type.value

        organization_type: str | Unset = UNSET
        if not isinstance(self.organization_type, Unset):
            organization_type = self.organization_type.value

        privacy_policy_url = self.privacy_policy_url

        registered_address: dict[str, Any] | Unset = UNSET
        if not isinstance(self.registered_address, Unset):
            registered_address = self.registered_address.to_dict()

        stock_exchange = self.stock_exchange

        stock_symbol = self.stock_symbol

        tax_id = self.tax_id

        tax_id_issuing_country = self.tax_id_issuing_country

        terms_and_conditions_url = self.terms_and_conditions_url

        website_url = self.website_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if authorized_contact_email is not UNSET:
            field_dict["authorized_contact_email"] = authorized_contact_email
        if authorized_contact_first_name is not UNSET:
            field_dict["authorized_contact_first_name"] = authorized_contact_first_name
        if authorized_contact_last_name is not UNSET:
            field_dict["authorized_contact_last_name"] = authorized_contact_last_name
        if authorized_contact_phone is not UNSET:
            field_dict["authorized_contact_phone"] = authorized_contact_phone
        if authorized_contact_title is not UNSET:
            field_dict["authorized_contact_title"] = authorized_contact_title
        if brand_name is not UNSET:
            field_dict["brand_name"] = brand_name
        if business_registration_country is not UNSET:
            field_dict["business_registration_country"] = business_registration_country
        if business_verification_documents is not UNSET:
            field_dict["business_verification_documents"] = (
                business_verification_documents
            )
        if industry is not UNSET:
            field_dict["industry"] = industry
        if legal_business_name is not UNSET:
            field_dict["legal_business_name"] = legal_business_name
        if legal_entity_type is not UNSET:
            field_dict["legal_entity_type"] = legal_entity_type
        if organization_type is not UNSET:
            field_dict["organization_type"] = organization_type
        if privacy_policy_url is not UNSET:
            field_dict["privacy_policy_url"] = privacy_policy_url
        if registered_address is not UNSET:
            field_dict["registered_address"] = registered_address
        if stock_exchange is not UNSET:
            field_dict["stock_exchange"] = stock_exchange
        if stock_symbol is not UNSET:
            field_dict["stock_symbol"] = stock_symbol
        if tax_id is not UNSET:
            field_dict["tax_id"] = tax_id
        if tax_id_issuing_country is not UNSET:
            field_dict["tax_id_issuing_country"] = tax_id_issuing_country
        if terms_and_conditions_url is not UNSET:
            field_dict["terms_and_conditions_url"] = terms_and_conditions_url
        if website_url is not UNSET:
            field_dict["website_url"] = website_url

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.postal_address import PostalAddress

        d = dict(src_dict)
        authorized_contact_email = d.pop("authorized_contact_email", UNSET)

        authorized_contact_first_name = d.pop("authorized_contact_first_name", UNSET)

        authorized_contact_last_name = d.pop("authorized_contact_last_name", UNSET)

        authorized_contact_phone = d.pop("authorized_contact_phone", UNSET)

        authorized_contact_title = d.pop("authorized_contact_title", UNSET)

        brand_name = d.pop("brand_name", UNSET)

        business_registration_country = d.pop("business_registration_country", UNSET)

        def _parse_business_verification_documents(
            data: object,
        ) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                business_verification_documents_type_0 = cast(list[str], data)

                return business_verification_documents_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        business_verification_documents = _parse_business_verification_documents(
            d.pop("business_verification_documents", UNSET)
        )

        industry = d.pop("industry", UNSET)

        legal_business_name = d.pop("legal_business_name", UNSET)

        _legal_entity_type = d.pop("legal_entity_type", UNSET)
        legal_entity_type: BusinessProfileRequestLegalEntityType | Unset
        if isinstance(_legal_entity_type, Unset):
            legal_entity_type = UNSET
        else:
            legal_entity_type = BusinessProfileRequestLegalEntityType(
                _legal_entity_type
            )

        _organization_type = d.pop("organization_type", UNSET)
        organization_type: BusinessProfileRequestOrganizationType | Unset
        if isinstance(_organization_type, Unset):
            organization_type = UNSET
        else:
            organization_type = BusinessProfileRequestOrganizationType(
                _organization_type
            )

        privacy_policy_url = d.pop("privacy_policy_url", UNSET)

        _registered_address = d.pop("registered_address", UNSET)
        registered_address: PostalAddress | Unset
        if isinstance(_registered_address, Unset):
            registered_address = UNSET
        else:
            registered_address = PostalAddress.from_dict(_registered_address)

        stock_exchange = d.pop("stock_exchange", UNSET)

        stock_symbol = d.pop("stock_symbol", UNSET)

        tax_id = d.pop("tax_id", UNSET)

        tax_id_issuing_country = d.pop("tax_id_issuing_country", UNSET)

        terms_and_conditions_url = d.pop("terms_and_conditions_url", UNSET)

        website_url = d.pop("website_url", UNSET)

        business_profile_request = cls(
            authorized_contact_email=authorized_contact_email,
            authorized_contact_first_name=authorized_contact_first_name,
            authorized_contact_last_name=authorized_contact_last_name,
            authorized_contact_phone=authorized_contact_phone,
            authorized_contact_title=authorized_contact_title,
            brand_name=brand_name,
            business_registration_country=business_registration_country,
            business_verification_documents=business_verification_documents,
            industry=industry,
            legal_business_name=legal_business_name,
            legal_entity_type=legal_entity_type,
            organization_type=organization_type,
            privacy_policy_url=privacy_policy_url,
            registered_address=registered_address,
            stock_exchange=stock_exchange,
            stock_symbol=stock_symbol,
            tax_id=tax_id,
            tax_id_issuing_country=tax_id_issuing_country,
            terms_and_conditions_url=terms_and_conditions_url,
            website_url=website_url,
        )

        business_profile_request.additional_properties = d
        return business_profile_request

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
