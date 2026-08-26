

# TTS improvements for go acceleration (acceleration folder)

## Expressiveness

When using ElevenLabs as a TTS. Expand the instructions to the LLM to say that it can use emotional TTS
See https://elevenlabs.io/blog/v3-audiotags

These tags are part of the tags that are supported. Don't use an ENUM, its AI driven, you can use any string. But the documented supported ones are

Emotions: [happy], [sad], [angry], [excited], [worried], [curious], [crying], [annoyed], [appalled], [thoughtful], [surprised], [mischievously], [sorrowful], [elated]
Delivery: [whispers], [shouts], [softly], [slow], [rushed], [drawn out], [cautiously], [cheerfully], [sarcastic], [muttering], [indecisive], [quizzically]
Human sounds: [laughs], [laughs harder], [starts laughing], [laughing], [chuckles], [giggling], [sighs], [exhales], [exhales sharply], [inhales deeply], [clears throat], [wheezing], [snorts], [groaning], [swallows], [gulps]
Pacing/conversation: [pause], [short pause], [long pause], [jumping in]
Character/style: [French accent], [strong French accent], [British accent], [pirate voice], [auctioneer], [sings]

Keep in mind that other TTS solutions will probably expose this in a different way. So the best solution here is to just extend
the instructions with some TTS specific usage tips.

## Custom voices

Allow CRUD for storing custom voices in our go backend. 
Add an endpoint that allows you to prepare a voice for a given TTS.
The implementation of cloning/setting up that voice for a given TTS provider is different for each one of them.

# STT

## Keyterms

Allow defining keyterms in the agent config (stored in the db). this is commonly used if there are some business specific terms you want the AI to handle.



