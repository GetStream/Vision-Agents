# Telnyx Phone Examples

Minimal inbound and outbound phone examples for the Telnyx plugin. These examples
use Telnyx Call Control, Telnyx Media Streaming, Stream, and Gemini Realtime.

## Requirements

Create a `.env` file at the repo root or export these variables:

```bash
STREAM_API_KEY=
STREAM_API_SECRET=
GOOGLE_API_KEY=
TELNYX_API_KEY=
TELNYX_PUBLIC_KEY=
```

`TELNYX_PUBLIC_KEY` is the Base64 Ed25519 public key from the Telnyx Mission
Control Portal. The examples verify webhook signatures before handling events.

You also need a Telnyx phone number. You can pass it to the examples with
`--from` or `--phone-number`, or set:

```bash
TELNYX_PHONE_NUMBER=+15551234567
```

Start ngrok before running either example:

```bash
ngrok http 8000
```

The examples auto-detect the local ngrok HTTPS tunnel. You can also set
`NGROK_URL=example.ngrok-free.app` or pass `--ngrok-url`.

## Quick Start

Use `--setup-telnyx` for the most direct local development flow. The example
creates a temporary Telnyx Call Control App with webhook URL
`https://<NGROK_URL>/telnyx/events` and deletes it on normal shutdown.

Outbound:

```bash
uv run plugins/telnyx/examples/outbound_call.py \
  --setup-telnyx \
  --from +15551234567 \
  --to +15557654321
```

Inbound:

```bash
uv run plugins/telnyx/examples/inbound_call.py \
  --setup-telnyx \
  --phone-number +15551234567
```

For inbound calls, `--setup-telnyx` also routes the Telnyx number to the
temporary Call Control App and restores the previous routing on normal shutdown.

Restricted Telnyx accounts may only call verified destination numbers. The
outbound example checks that by default. If your account is unrestricted, you can
skip that preflight:

```bash
uv run plugins/telnyx/examples/outbound_call.py \
  --setup-telnyx \
  --from +15551234567 \
  --to +15557654321 \
  --skip-verified-destination-check
```

## Manual Telnyx Setup

If you do not want the examples to create or route Telnyx resources, configure
Telnyx yourself and omit `--setup-telnyx`.

1. Create a Telnyx Call Control App.
2. Set the app webhook URL to:

   ```text
   https://<NGROK_URL>/telnyx/events
   ```

3. For inbound calls, route your Telnyx phone number to that Call Control App.
4. For outbound calls, make sure the Call Control App has an outbound voice
   profile that can call your target country.
5. If your Telnyx account is restricted, verify the destination phone number in
   Telnyx before running the outbound example.

Then set:

```bash
TELNYX_CALL_CONTROL_APP_ID=
TELNYX_PHONE_NUMBER=+15551234567
NGROK_URL=example.ngrok-free.app
```

Inbound manual setup also needs the Telnyx phone number resource ID:

```bash
TELNYX_PHONE_NUMBER_ID=
```

Run manually configured outbound:

```bash
uv run plugins/telnyx/examples/outbound_call.py --to +15557654321
```

Run manually configured inbound:

```bash
uv run plugins/telnyx/examples/inbound_call.py
```

## Why Call Control Is Required

Telnyx media streaming for programmable calls is configured through a Call
Control App. A regular phone-number connection, including a forwarding-only
connection, is not enough for these examples because the app needs a webhook URL
where Telnyx can send call events and receive answer/dial commands.

For outbound calls, the Call Control App ID is passed to the Telnyx Dial API as
`connection_id`.

For inbound calls, the Telnyx phone number must be routed to the same Call
Control App so Telnyx sends `call.initiated` webhooks to this example.

## Common Setup Errors

- `Invalid value for connection_id`: the Call Control App ID is missing,
  invalid, or inactive.
- Webhook URL mismatch: update the Call Control App webhook URL to the current
  ngrok URL, or use `--setup-telnyx`.
- Inbound number not routed: assign the Telnyx phone number to the Call Control
  App, or use `--setup-telnyx`.
- Destination not verified: verify the `--to` number in Telnyx or use an
  unrestricted account.
