---
name: pricing
description: How to make new billable work (a model, vendor, modality or phone feature) report usage and get priced correctly. Read before adding anything the router bills, or when work shows up unpriced.
---

# Pricing and usage

Providers report quantities; the price list turns them into cost. Never compute cost in a provider.

- **Meters:** `internal/pricing/pricing.go`. **List prices:** `acceleration/prices/vendor.json` (rates only, `region` required).
- **The database is the source of truth.** On start a router adds items the database has never priced, at rates × 1.08. It never changes an existing price, so get the rates right before the first deploy. Later changes go through a price change per environment (management API) plus a `vendor.json` PR. See `check_prices`.
- **Unpriced work is recorded at $0.** It reserves no spend and raises `unpriced`, and configs naming it get a 400. `free: true` is a price; a missing entry is not.
- **Every meter the vendor bills needs a rate.** Missing cached or cache-write rates fall back to input. A missing `output_tokens` rate does not fall back: it is free.
- **Vendor cost:** if the vendor's response says what it charged, put it in `VendorCostMicros` so `router prices drift` can check the rate.

## Every change

1. Report usage from the provider, in meters the vendor bills.
2. Add the `vendor.json` entry from the vendor's pricing page, dated in the `router.yaml` comment. Self-hosted models get `local: true` instead.
3. Run `go test ./internal/routing/ -run TestEveryRoutedModelIsPriced`, `router prices check`, and the modality's integration suite.
4. After the deploy, in each environment: `router prices show --unpriced` is empty, the start log has no "billable work has no price", and one real request returns `X-Stream-Cost-Micros` equal to quantity × price.

## By type

- **LLM:** `input_tokens`, `cached_input_tokens`, `cache_write_input_tokens`, `output_tokens`. Write a long-prompt surcharge as `tiers` (`above_input_tokens`) and a service tier as `when: {service_tier}`. A customer-key (BYOK) twin is generated: never write one. See `router-llm`.
- **STS:** split by modality (`input_audio_tokens`, `output_text_tokens`, …), plus generic `input_tokens`/`output_tokens` at the audio rates as a fallback for vendors that report no split. See `router-sts`.
- **STT:** `audio_ms` (per minute is `per: 3600000`). See `router-stt`.
- **TTS:** `characters`, or `audio_ms` if the vendor bills by duration. See `router-tts`.
- **Decision model:** `input_tokens`, plus `output_tokens` if billed. A System One vendor reports usage for you; `:free` endpoints use `free: true`. See `decision_model`.
- **Search:** `requests`, or tokens for LLM-backed search. See `router-search`.
- **Image, detection, OCR:** `images`, `pages`, or tokens, whichever the vendor bills. See `router-detection`.
- **Memory, knowledge:** `requests`; knowledge also uses `bytes_written`, `bytes_queried`, `bytes_returned`, `results` and `byte_hours`. Add new billed work to `internal/billable` so `--unpriced` checks it.
- **Sandbox:** `vcpu_seconds`, `memory_gb_seconds` and `disk_gb_seconds`, with `when: {class}`, recorded by the usage jobs. Each new region or target needs entries.
- **Phone:** per vendor, `dial` and `call_minutes` (`call_seconds`); SMS uses `messages`, `message_segments` and `media_messages` with `when` by `country`, `direction` or `category`. A cost the vendor quotes outright (a number's month) goes through `Quoted`, with no entry. WhatsApp and iMessage are billed by the app's own account: don't price them. A new vendor goes in `billable.Items`. See `telnyx`, `phonecheck`.
- **Runtime:** `stream` `text`/`voice`/`video` by `seconds`, recorded in slices by `routing.NewRecorder`. These are the operator's own prices: set them by price change, not in `vendor.json`.
