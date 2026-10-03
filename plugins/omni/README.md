# Omni Plugin

Slack, WhatsApp, RCS, SMS and iMessage (through [Linq](https://linqapp.com)) messages as
Stream Chat messages, so an agent answers every channel from one Stream channel.

Each channel module reads its provider's webhook body into an `OmniMessage` with
`parse`, and turns an `OmniMessage` into the request bodies its provider's send API takes
with `render`. `to_stream` and `from_stream` convert between an `OmniMessage` and a Stream
message.

| Module     | `parse` reads                         | `render` bodies are sent to                          |
| ---------- | ------------------------------------- | ---------------------------------------------------- |
| `slack`    | Events API `event_callback`           | `chat.postMessage`                                   |
| `whatsapp` | Cloud API webhook                     | `POST /{phone-number-id}/messages`                   |
| `rcs`      | Google RBM webhook or Pub/Sub push    | `POST /v1/phones/{phone}/agentMessages?messageId=…`  |
| `sms`      | Twilio incoming message webhook form  | `POST /2010-04-01/Accounts/{sid}/Messages.json` form |
| `linq`     | Linq v3 webhook, version `2026-02-03` | `POST /api/partner/v3/chats/{chat_id}/messages`      |

## Installation

```bash
uv add "vision-agents[omni]"
```

## Usage

```python
from vision_agents.plugins import omni

for message in omni.whatsapp.parse(webhook_body):
    await client.upsert_users(
        UserRequest(id=omni.stream_user_id(message), name=message.sender_name or None)
    )
    channel = client.chat.channel("messaging", omni.stream_channel_id(message))
    await channel.send_message(omni.to_stream(message))

# Later, the agent's reply goes back where the message came from.
reply = omni.from_stream(agent_message, route=message)
for body in omni.whatsapp.render(reply):
    ...  # POST it with the account's token
```

In Stream a message keeps its text as the message text and its files as `image`, `video`,
`audio` and `file` attachments, with `mime_type`, `file_size` and the provider's
`media_id` on them. A shared place is a `location` attachment with `latitude` and
`longitude`. Who sent it, the conversation and the account it is on are in
`custom.omni`, which `from_stream` reads back when no `route` is given.

Webhook signatures are not checked here, and media URLs from Slack, Twilio and RCS need
the account's credentials to download. WhatsApp media arrives as a `media_id` only.
