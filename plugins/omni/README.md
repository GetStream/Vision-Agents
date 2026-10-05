# Omni Plugin

Slack, Teams, WhatsApp, RCS, SMS and iMessage messages as Stream Chat messages, so an agent
answers every channel from one Stream channel per conversation.

A channel can be carried by more than one provider. Each provider reads its webhook bodies
into `OmniMessage`s with `parse`, and turns an `OmniMessage` into the request bodies its
send API takes with `render`. A `ProviderRegistry` finds the provider by name, and every
message records the provider it came through.

| Provider     | Channels              | `parse` reads                                | `render` bodies are sent to                          |
| ------------ | --------------------- | -------------------------------------------- | ---------------------------------------------------- |
| `slack`      | Slack                 | Events API `event_callback`                  | `chat.postMessage`                                   |
| `teams`      | Teams                 | Bot Framework `message` activity             | `POST {serviceUrl}/v3/conversations/{id}/activities` |
| `whatsapp`   | WhatsApp              | Cloud API webhook                            | `POST /{phone-number-id}/messages`                   |
| `google_rbm` | RCS                   | Google RBM webhook or Pub/Sub push           | `POST /v1/phones/{phone}/agentMessages?messageId=…`  |
| `twilio`     | SMS, WhatsApp, RCS    | Twilio incoming message webhook form         | `POST /2010-04-01/Accounts/{sid}/Messages.json` form |
| `telnyx`     | SMS                   | Telnyx `message.received` webhook            | `POST /v2/messages`                                  |
| `linq`       | iMessage, RCS, SMS    | Linq v3 webhook, version `2026-02-03`        | `POST /api/partner/v3/chats/{chat_id}/messages`      |

## Installation

```bash
uv add "vision-agents[omni]"
```

## Usage

```python
from vision_agents.plugins import omni

providers = omni.ProviderRegistry()

for message in providers.parse(omni.Provider.LINQ, webhook_body):
    await client.upsert_users(
        UserRequest(id=omni.stream_user_id(message), name=message.sender_name or None)
    )
    channel = client.chat.channel("messaging", omni.stream_channel_id(message))
    await channel.send_message(omni.to_stream(message))

# Later, the agent's reply goes back the way the message came.
reply = omni.from_stream(agent_message, route=message)
for body in providers.render(reply):
    ...  # POST it with the provider's credentials
```

Keep one registry for the life of the app: the Slack provider remembers the messages it
has read, so a message Slack delivers as both `message` and `app_mention`, or retries, is
read once.

To add a provider, subclass `OmniProvider` with a unique `name`, the `channels` it carries,
`parse` and `render`, and `register` it.

## In Stream

A message keeps its text as the message text and its files as `image`, `video`, `audio`
and `file` attachments, with `mime_type`, `file_size` and the provider's `media_id` on
them. A shared place is a `location` attachment with `latitude` and `longitude`. Its
channel, provider, sender, conversation and account are in `custom.omni`, which
`from_stream` reads back when no `route` is given.

Stream ids are digests, so they fit Stream's limits whatever the provider's ids are:

- `stream_channel_id`: `{provider}_{digest of account and conversation}`, one Stream
  channel per conversation with each of the business's numbers or accounts.
- `stream_user_id`: `{provider}_{digest of sender}`.

Webhook signatures are not checked here, and media URLs from Slack, Twilio and RCS need
the provider's credentials to download. WhatsApp media arrives as a `media_id` only.
