---
name: llm-interface
description: Consider changes to the llm standardization/interface
---

# LLM Interface

Start by reviewing the latest changes here:
- OpenAI: https://developers.openai.com/api/docs/guides/latest-model
- Claude: https://platform.claude.com/docs/en/build-with-claude/working-with-messages
- Gemini: https://ai.google.dev/gemini-api/docs/latest-model
- Grok: https://docs.x.ai/developers/model-capabilities/text/generate-text

## What to standardize

The feature set for voice AI definitely needs to be standardized. 
- Text input into streaming response
- Usage / cost tracking
- Cache control
- Conversation persistence (use store:true or similar systems when available)
- Tool calling

This requires text response creation to be standardized

But in general it's good to standardize more. This makes it easier to switch between models

## What not to standardize

- Reasoning levels are different per model. They also don't map cleanly to each other. So lets just keep track for each LLM what's valid

## Cache Standards



## Guidelines

You can find the llm.go LLM interface
We mostly want to follow OpenAI's naming conventions where possible since they are the biggest in non-coding AI usage.

For openAI (but probably not for others) there are performance benefits to the websocket mode
https://developers.openai.com/api/docs/guides/websocket-mode


### History



### Cache control

CachePolicy:
- TTL (0 for no caching)

Breakpoints: 
- For models that support this we should automatically handle the setting of a break point after the instructions
- With a shared per agent config cache (so instructions are reused)
- While having it set to implicit mode so the customer conversation grows