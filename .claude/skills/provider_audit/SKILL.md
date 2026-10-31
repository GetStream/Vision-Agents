---
name: provider_audit
description: Audit what is left on every vendor account the product pays for — prepaid balance, credits, character or minute quota, spend this month — read-only, and say which account is about to run out. Use when asked how much money is left, to audit providers, for a vendor's balance, credits or quota, or whether any account is about to run out.
---

# Audit vendor accounts

A prepaid vendor account at zero refuses requests: the Gemini API, for one, answers
HTTP 402 to every key of the billing account
([billing](https://ai.google.dev/gemini-api/docs/billing)). The keypool alerts only fire
once calls fail. This skill reads what is left before that.

`vendors.md` is the inventory: every vendor the product depends on, what it is for, who
pays, which key the router reads, and how to see what is left. `audit.py` calls every
read-only balance API in it and prints one table.

Vendors are core or connector. Core vendors are the ones the platform needs to work
(models, memory, sandboxes, GPU hosts, infra); connector vendors (Telnyx, Twilio, Linq and
the other phone and channel vendors) are bring-your-own, used only by connectors.

## Run it

Everything is read-only: GETs to documented billing or usage endpoints. Never top up,
change a plan or a key from here.

The script reads keys only from the repo-root `.env`, run from the repo root:

```bash
uv run --no-project --env-file .env .claude/skills/provider_audit/audit.py --min USD=20
```

It prints `vendor | left | unit | source | checked_at | note`, one row per key (a pool
`NAME=k1,k2` gives two): the core rows, then the connector rows under their own heading.
Then `BELOW:` for each core balance under its `--min`, and `BELOW (connector):` for each
connector one. `--min` takes `UNIT=AMOUNT` and repeats (`--min USD=20 --min
characters=100000`); it compares only rows in that unit, so `USD spent this month` is
never a balance. Exit code 1 means a core balance is below its minimum. A connector
balance below it and an error row (a 401, a timeout, an odd body) are reported and never
fail the run.

For example:

```text
vendor | left | unit | source | checked_at | note
ExampleVendor | <amount> | USD | https://example.com/docs/balance | <time> |
OtherVendor | error | | https://example.com/docs/usage | <time> | error (OTHER_API_KEY): HTTP 401 ...
```

Read each row:

- a number: what is left, or spent this month for postpaid accounts;
- `error (ENV): HTTP 401 ...`: the key cannot read billing: a scope the API needs
  (`vendors.md` names it, with the doc) or a dead key;
- `skipped: ENV not set`: no such key in `.env`;
- `unknown (dashboard only)`: no API; open the URL in `source`.

Some balances need an admin or management key the router never holds:
`OPENAI_ADMIN_KEY`, `ANTHROPIC_ADMIN_KEY`, `DEEPGRAM_ADMIN_KEY`,
`OPENROUTER_MANAGEMENT_KEY`, `XAI_MANAGEMENT_KEY`. Set one in `.env` to read that
account; without it the row says `skipped: <VAR> not set`.

## Keep the inventory current

Each time, from the repo root:

```bash
# Every key the router code reads. Skip our own (ROUTER_*, USAGE_*, STREAM_*, CLIENT_SECRET,
# OTHER_TOKEN) and the connector tokens (GITHUB_, GITLAB_, LINEAR_). A second credential of
# a listed vendor (SINCH_APPLICATION_*, VONAGE_API_SECRET, VONAGE_PRIVATE_KEY,
# TELNYX_PUBLIC_KEY) needs no row of its own; *_PERSON_* keys are in the Telnyx and Twilio rows.
git grep -hoE '"[A-Z][A-Z0-9_]*_(API_KEY|TOKEN|SECRET|KEY)"' origin/accelerate -- \
  'acceleration/*.go' ':!*_test.go' | sort -u
# Every provider routed, per modality.
awk '/^[a-z_]+:/{s=$1} /^    - provider:/{print s, $3}' acceleration/internal/routing/router.yaml | sort -u
# Every implemented phone vendor and its credentials; each needs a row in vendors.md.
awk '/- vendor:/{v=$3} /implemented: true/{i=v} /credentials:/&&i==v{print v, $0}' acceleration/internal/phone/phone.yaml
```

Then the hosts that hold no router key: `acceleration/deploy/*/README.md` (GPU hosts),
`compose.yaml`, `.github/workflows/`, the deploy skill, and the clients' own repos
(Volt, the docs site). A new vendor gets a row in `vendors.md`. One with a read-only
balance API also gets a check in `audit.py`: a function under `@check("Vendor", "ENV")`
whose docstring is the API's doc URL, and a test in `test_audit.py`. A connector
vendor's name also goes in `CONNECTORS`. Cite the doc or
the live answer for every claim; mark the rest `unverified`.

The tests run a real local HTTP server in place of each vendor:

```bash
uv run --no-project --with pytest pytest -c /dev/null \
  --confcutdir .claude/skills/provider_audit .claude/skills/provider_audit
```
