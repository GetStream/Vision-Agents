

# TTS routing

TTS routing should support the following

tts:
    providers: ["elevenlabs-conversational-v3"],
    voice_id: "custom:123",
    data_policy:
        allow_training: false
        retention: none
    overwrites:
        elevenlabs:
            voice_id: myvoiceid
        inworld:
            deliveryMode


## Backend changes

A voice sample of 30s is stored on our end.

To sync that to a provider we need to track a voice_sync table
- when it was synced
- to which provider
- what the id the provider is using for our uploaded voice
- if sync succeeded or failed
