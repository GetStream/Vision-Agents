# agent_moderator

The starting point for an app's Nora: the agent the chat review worker hands flagged content to
through `POST /v1/agents/runs`. The dashboard's "AI moderator" template
(`volt-dashboard` `agent-templates.ts`) is meant to create the same agent; change both together.

- `instructions.md` is how Nora reviews: content is untrusted data, flags are hypotheses, the
  current content is judged first, and evidence is quoted exactly.
- `skills/` holds one skill per label family: profanity, harassment, suggestive language, and
  how to talk about actions.

The app may edit all of it. The answer format the worker parses is not here: the router adds it
(`acceleration/internal/moderator/contract.md`) after every other instruction on each run of a
config whose `role` is `agent_moderator`, so an edit cannot break parsing. A session that is not
a run, such as Ask Nora, does not get it and answers in plain sentences.
