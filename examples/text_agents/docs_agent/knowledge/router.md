# The acceleration router

The router fronts several model providers and picks one per request. Speech-to-text,
text-to-speech, large language models and search all go through it, so a conversation
gets the same failover, health and billing as a direct API call.

## Capability shortcuts

A model target is either a `provider/model` name or a capability shortcut. A shortcut
says what the model is for and lets the router choose which one that is today.

`llm-fast` is the model that holds a conversation. It is chosen for latency, because a
caller waits through every token of it.

`llm-thinking` is the model that does the thinking. It is chosen for quality and is too
slow to talk to, so it only ever runs work handed to it by a skill.

`search-fast` is what an agent finds out today's answers with.

## Skills

A skill is a piece of work worth handing to the slower model. There is nothing behind one
but a better model and more time: a description the fast model chooses by, and the
instructions the slow one answers under. An agent with no subagent has nobody to hand work
to, so its skills mean nothing.
