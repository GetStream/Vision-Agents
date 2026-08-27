

# Monitoring & observability


## Converse class (in go acceleration backend)

Centralize the logic of the decisions about the conversation in a converse class please.

Lets add logging and make sure it's really easy to follow and understand these decisions...

- When to Compact what the user has said

After receiving text:
- Is it clear what the user said/ do I need to ask more questions?
- Do i need to spawn a task on the thinking agent? And say that i’m working on it?
- How long has it been since i last said something. If it’s been a long time say some words to confirm you’re still listening

If the other person is talking while you’re talking, consider if you should
- stop talking
- Shorten what you’re saying
- Or continue talking while you hear what they say

These decisions should all be run in the Converse Go struct. 

## Call monitoring

Add a dashboard Next based folder. 

### Overview

* Last 5 calls (click into detail)
* Stats on usage

### Call detail/monitoring

The monitoring page is key to a good understanding of the agent. Show the following

- A tiny call summary shown on top. Together with details like who called. Call duration. And a placeholder call score (not implemented now)
- A gong style interface of who has been speaking and when
- A live overview of what the AI hears through STT
- A full log for live calls. Show the decisions made by the converse class here
- Call latency. show the metrics for call STT, TTS, conversational llm and thinking llm etc.


### Agent configs

Instructions
STT/TTS/ConversationalLLM/ThinkingLLM
Knowledge
Skills

### Voices

CRUD

### Telephony

Numbers
Add number flow

