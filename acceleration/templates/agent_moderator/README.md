# agent_moderator

The starting point for an app's Nora: the agent the chat review worker hands flagged content to
through `POST /v1/agents/runs`. The dashboard's "AI moderator" template
(`volt-dashboard` `agent-templates.ts`) is meant to create the same agent; change both together.

- `instructions.md` is how Nora reviews: content is untrusted data, flags are hypotheses, the
  current content is judged first, and evidence is quoted exactly.
- `skills/` holds one skill per label family: profanity, harassment, suggestive language, and
  how to talk about actions.
- `simulations/` holds the reviews Nora must get right: casual against targeted profanity,
  educational quotation, repeat and overturned history, an innocent message from a banned user,
  an injected instruction, and two Ask Nora questions across the app (who violates most, which
  label is most often overturned). Run them after editing the agent
  (`getstream agents simulate nora`); the AI team runs them before changing this template.
- `skills/suggest_policy.md` is for Ask Nora: from the live policy, the last 7 days of reviews
  and where Nora disagreed with the first layer, it proposes changes to labels, app context,
  severities, blocklists, rules and the flooding filter, as JSON the dashboard's
  `apply-configuration.ts` saves once the moderator confirms. It only proposes. The dashboard
  calls it by name (`suggest-policy-skill.ts`), so keep the file name.

The app may edit all of it. The answer format the worker parses is not here: the router adds it
(`acceleration/internal/moderator/contract.md`) after every other instruction on each run of a
config whose `role` is `agent_moderator`, so an edit cannot break parsing. A session that is not
a run, such as Ask Nora, does not get it and answers in plain sentences.
