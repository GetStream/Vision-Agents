
## Sandbox

The hosted version ships with a Sandbox where you can try things with very low volume

Limits can change, but for now are set at

* outbound text and voice to 2 phone numbers
* at most 30 messages daily or 30m of audio

This is just in place for testing purposes, for production traffic 10 DLC setup is needed


## 10 DLC

When customers use either
- RCS
- Text outbuond
- Voice outbound
- Whatsapp
- iMessage

We need to require them to complete a 10DLC setup


## 10DLC use case submission logs

For each app a customer can submit one or multiple 10DLC use cases.
An agent can either use the default or a custom 10DLC use case.

The full list of fields we need to store is shown below. 
In addition that we need tracking fields

10dlc_use_case with created_at, updated_at, submitted_at, approved_at
10dlc_review_log (separate table with review notes, created_at, submitted_at, approved_at etc)

## User opt out

It should also be possible for users to opt out.
So add an extra table for that

## 10DLC submissions

After we check, the submission will go to one of the telephony vendors (twilio, telnyx etc)
So also add API support to sync it

## Dashboard

Add a page to the dashboard for submitting and viewing your 10DLC use cases
Also update the docs with a page about 10DLC in the telephony section


# 10DLC fields

# Stream channel onboarding fields

Scope: US voice and RCS through Telnyx, WhatsApp Business Platform, and iMessage through Linq. Based on documentation reviewed October 5, 2026.

**Required** means required for the specified flow. **Conditional** depends on the customer, country, or use case. **Recommended** means a Stream product requirement rather than a published provider requirement. Capture provider-generated IDs automatically.

## Shared / all use cases

Maintain one reusable business profile. These fields are **not universally mandatory on every channel**: 10DLC and RCS require more business information than basic voice or Linq. Approval remains separate for each channel.

- [ ] `legal_business_name`
- [ ] `brand_name` — Customer-facing name.
- [ ] `legal_entity_type` — Corporation, LLC, partnership, sole proprietor, etc.
- [ ] `organization_type` — Private/public company, nonprofit, government.
- [ ] `business_registration_country`
- [ ] `tax_id` / `business_registration_id` — Conditional on channel and entity.
- [ ] `tax_id_issuing_country` — Conditional.
- [ ] `registered_address` — Street, city, state/region, postal code, country.
- [ ] `website_url`
- [ ] `industry`
- [ ] `authorized_contact_first_name`
- [ ] `authorized_contact_last_name`
- [ ] `authorized_contact_title`
- [ ] `authorized_contact_email`
- [ ] `authorized_contact_phone`
- [ ] `privacy_policy_url` — Required for RCS and relevant messaging approval flows.
- [ ] `terms_and_conditions_url` — Required for RCS and relevant messaging approval flows.
- [ ] `stock_symbol` / `stock_exchange` — Conditional: public companies and provider requirements.
- [ ] `business_verification_documents` — Conditional: requested identity/business evidence.

Sources: [Telnyx 10DLC overview](https://telnyx.com/resources/what-is-10dlc), [Telnyx RCS onboarding](https://support.telnyx.com/en/articles/16624885-rcs-api-onboarding-guide), [Meta verification documents through Telnyx](https://support-v2.telnyx.com/en/articles/16300025-whatsapp-documents-accepted-for-meta-business-verification).

## Outbound voice AI

### Required for caller identity

- [ ] `outbound_caller_id_number` — Telnyx-provisioned/ported number, or verified external number.
- [ ] `caller_id_verification` — Conditional: external number; SMS/voice challenge or approved bulk verification.
- [ ] `caller_id_verification_status` — Automatic.

External-number verification is separate from SMS registration. Voice does not require a 10DLC campaign.

Source: [Telnyx verified numbers FAQ](https://support-v2.telnyx.com/en/articles/6790265-verified-numbers-faq).

### Recommended Stream onboarding fields

These are not a standardized Telnyx registration form.

- [ ] `calling_purpose` — Support, reminders, sales, etc.
- [ ] `destination_countries`
- [ ] `expected_call_volume`
- [ ] `consent_collection_method`
- [ ] `consent_disclosure_text`
- [ ] `consent_evidence_location`
- [ ] `opt_out_handling`
- [ ] `call_recording_enabled`
- [ ] `recording_disclosure_and_consent_method` — Conditional: recording.

### Per-recipient records

Suggested record fields for demonstrating permission, rather than company-onboarding fields:

- [ ] `recipient_phone`
- [ ] `consent_scope`
- [ ] `consent_timestamp`
- [ ] `consent_evidence`
- [ ] `consent_revoked_at`

Covered AI calls require consent; covered telemarketing calls require written consent, subject to exceptions. Business verification alone does not provide permission to call.

Source: [FCC ruling on AI-generated voices](https://www.fcc.gov/document/fcc-confirms-tcpa-applies-ai-technologies-generate-human-voices).

## WhatsApp

### Required setup

- [ ] `meta_business_portfolio` — Select existing or create through Meta.
- [ ] `whatsapp_business_account` — Select existing or create.
- [ ] `meta_authorization` — Customer grants the integration access.
- [ ] `whatsapp_phone_number`
- [ ] `whatsapp_display_name`
- [ ] `phone_verification_method` — SMS or voice, where required.
- [ ] `phone_verification_completed`

### Capture automatically

- [ ] `meta_business_id`
- [ ] `waba_id`
- [ ] `phone_number_id`
- [ ] Access credentials and permission status.

Obtain these through Embedded Signup and the associated API flow.

### Conditional

- [ ] Business verification documents, if Meta requests them.
- [ ] Message template name, language, category, content, and variable examples — For messages requiring approved templates.
- [ ] Billing setup — Depending on whether the customer or provider pays Meta.

Sources: [Telnyx WhatsApp setup](https://support.telnyx.com/en/articles/13986485-how-to-set-up-whatsapp-on-telnyx), [Meta Embedded Signup](https://www.postman.com/meta/whatsapp-business-platform/documentation/du6gzjv/embedded-signup), [WhatsApp templates](https://support-v2.telnyx.com/en/articles/13986486-how-to-create-whatsapp-message-templates), [Business verification documents](https://support-v2.telnyx.com/en/articles/16300025-whatsapp-documents-accepted-for-meta-business-verification).

## RCS

### Required business profile

Use the shared legal name, brand name, entity/organization types, website, EIN, registered address, and authorized contact details. The verification contact must use a **named individual's company-domain email**, not a generic mailbox or free email account. Public companies additionally provide their stock symbol.

### Required agent profile

- [ ] `display_name` — Maximum 40 characters.
- [ ] `description` — Maximum 100 characters.
- [ ] `use_case` — OTP, transactional, promotional, or multi-use.
- [ ] `logo_url` — 224 × 224 PNG/JPEG, maximum 50 KB.
- [ ] `hero_url` — 1440 × 448 PNG/JPEG, maximum 200 KB.
- [ ] `brand_color` — Hex color; at least 4.5:1 contrast against white.
- [ ] `privacy_policy_url`
- [ ] `terms_and_conditions_url`
- [ ] `support_phone` **or** `support_email`, with a display label.

Use public HTTPS URLs for assets and legal pages.

### Required launch-review information

- [ ] `company_overview`
- [ ] `agent_overview`
- [ ] `interaction_types`
- [ ] `message_examples`
- [ ] `opt_in_methods`
- [ ] `call_to_action_text`
- [ ] `call_to_action_url` / supporting media — As applicable to the consent flow.
- [ ] `double_opt_in` — Whether used.
- [ ] `opt_in_confirmation_message`
- [ ] `help_response`
- [ ] `opt_out_response`
- [ ] `test_video_url` — Demonstrates consent confirmation, example interactions, HELP, and STOP.

### Capture automatically

- [ ] Brand ID.
- [ ] Agent ID.
- [ ] Messaging profile association.
- [ ] Verification and launch statuses.

Sources: [Telnyx RCS onboarding](https://support.telnyx.com/en/articles/16624885-rcs-api-onboarding-guide), [RCS registration quickstart](https://developers.telnyx.com/docs/messaging/rcs/agent-registration), [RCS launch API](https://developers.telnyx.com/api-reference/rcs-agents/submit-an-rcs-agent-for-launch).

## Linq / iMessage

### Documented technical prerequisites

- [ ] Linq account/API access.
- [ ] Provisioned sending phone number.
- [ ] API bearer token — Managed by Stream if Stream owns the integration.
- [ ] Recipient phone number in E.164 format — Per conversation.

### Optional branding

- [ ] Contact-card name.
- [ ] Contact-card profile image — Square, minimum 200 × 200.
- [ ] Additional contact-card details.

### Recommended Stream fields

- [ ] Intended messaging use case.
- [ ] Customer-to-sending-number assignment.
- [ ] Consent collection method and evidence.
- [ ] Opt-out handling.
- [ ] Allowed fallback channels: iMessage, RCS, SMS.

**Unconfirmed:** Linq's public documentation does not provide an exhaustive downstream-customer KYC/registration schema. Confirm its reseller requirements directly before treating the shared business profile as sufficient. Linq iMessage onboarding is distinct from Apple Messages for Business and from Telnyx RCS Business Messaging registration.

Sources: [Linq quickstart](https://docs.linqapp.com/channel/imessage/getting-started/quickstart/), [Linq FAQ](https://docs.linqapp.com/channel/imessage/guides/resources/faq/).
