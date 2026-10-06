-- +goose Up

-- business_profiles is who an app is, written once and reused by every channel it registers
-- for: 10DLC, RCS and WhatsApp all ask for the same legal name, tax id and contact. One row
-- per app. vendor_brand_id is the brand a telephony vendor registered it as, once it has.
CREATE TABLE business_profiles (
    customer_id TEXT PRIMARY KEY,
    legal_business_name TEXT NOT NULL DEFAULT '',
    brand_name TEXT NOT NULL DEFAULT '',
    -- legal_entity_type is corporation, llc, partnership, sole_proprietor or other, and
    -- organization_type private, public, nonprofit or government.
    legal_entity_type TEXT NOT NULL DEFAULT '',
    organization_type TEXT NOT NULL DEFAULT '',
    business_registration_country TEXT NOT NULL DEFAULT '',
    tax_id TEXT NOT NULL DEFAULT '',
    tax_id_issuing_country TEXT NOT NULL DEFAULT '',
    registered_address JSONB NOT NULL DEFAULT '{}',
    website_url TEXT NOT NULL DEFAULT '',
    industry TEXT NOT NULL DEFAULT '',
    authorized_contact_first_name TEXT NOT NULL DEFAULT '',
    authorized_contact_last_name TEXT NOT NULL DEFAULT '',
    authorized_contact_title TEXT NOT NULL DEFAULT '',
    authorized_contact_email TEXT NOT NULL DEFAULT '',
    authorized_contact_phone TEXT NOT NULL DEFAULT '',
    privacy_policy_url TEXT NOT NULL DEFAULT '',
    terms_and_conditions_url TEXT NOT NULL DEFAULT '',
    stock_symbol TEXT NOT NULL DEFAULT '',
    stock_exchange TEXT NOT NULL DEFAULT '',
    business_verification_documents TEXT[] NOT NULL DEFAULT '{}',
    vendor_brand_id TEXT NOT NULL DEFAULT '',
    brand_status TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- +goose Down
DROP TABLE business_profiles;
