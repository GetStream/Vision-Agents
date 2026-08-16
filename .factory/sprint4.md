
## Cost tracking

We want to ensure that all requests

- STT, TTS, LLM routing
- Full agent workflow
- Phone numbers all have correct cost tagging

This is very useful when you have multiple customers and want to understand what drives costs.

So all API calls should support cost tags like this

cost_tracking={customer_id: 123, project: moderation, environment: dev}

Where customer_id, project, environment are all examples of keys the customer can provide. 
they can send whatever key they want. this is not typed, just a map of labels/tags

## Message storage

Store the chat or transcription messages into a stream chat channel based on the agent id

## Observability

For each call/session we want to track
- response times on STT, TTS and LLM
- full roundtrip delay (typically STT -> TTS -> LLM but can be more complicated with realtime LLM or openAI style voice)
- in the future we also want to track the delay from voice in/voice out
- time to first reply/token

## Phone numbers

Add a standardization layer for phone. We want to support these vendors

Twilio
Sinch
Telnyx
Bandwidth
BICS
Infobip
Vonage
Tata Communications / Kaleyra
Bird
DIDWW
Plivo

Across these vendors we want to support

- Searching for a phone number
- Buying a phone number
- Connecting the stream call to SIP (inbound and outbound)
- see https://getstream.io/video/docs/api/sip/inbound-trunk/

## Memory

Implement mem0 to make sure the agents have memory. As a default keep the memory by
- The API key/ app_id that's using the backend + the customer_id
- For calls without auth don't store memory for now
