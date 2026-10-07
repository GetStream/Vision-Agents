# The audit log, and the sync that does not write over you

## Asked for

`getstream agents sync my-agent` wrote over whatever the dashboard held. There was no
revision, no etag and no history: `syncAgent` short-circuited only when the directory's
hash matched the one it stored last, and a dashboard write cleared that hash, so the next
sync always wrote. Somebody editing an agent's instructions in the browser lost them the
next time a colleague ran a sync, and nothing anywhere said who had done either.

Three things were asked for, in one piece because they are one piece:

- A record of every configuration change and **who made it, when**, readable by more than
  the agents feature.
- A screen for it in the dashboard: a page of its own, and the last few changes on the
  agent's own page.
- A sync that, when the dashboard has moved since the last one, **stops, shows what
  changed, and offers a choice**: write those changes into the local files, or replace
  them.

## What is audited, and what is not

The line is how often a thing changes, not how important it is.

**Audited — configuration, which somebody sets up and then rarely touches:** agent
configs, skills, knowledge documents, knowledge urls, router configs, plugin credentials,
policies.

**Not audited — what an agent does while it runs:** sessions, simulations, simulation
runs, calls, invocations, transcripts, logs.

A session is read from the sessions list, a call from its record, a run from its run. Those
are traffic: one agent can produce thousands a day, and a log that filled with them would
be a worse sessions list rather than a history anybody could read. The audit log answers
"who changed this, and to what", which is only a question worth asking about something that
was supposed to stay still.

The constraint is in the schema, not in a convention: `audit_log.resource_type` is a CHECK
over exactly those seven, so a write recording a session is rejected by Postgres.

## What exists

### The table

[migrations/20261008120000_audit_log.sql](../../acceleration/migrations/20261008120000_audit_log.sql).
One row per change: the resource and what it was called at the time, the agent it belongs
to (blank for a router, which belongs to none), the action, the client it came from, who
was at the keyboard, the request id, and the fields that moved as a JSONB array of
`{field, before, after}`. Three indexes, each ending in `(created_at DESC, id DESC)`, which
is the sort the cursor pages by.

`resource_type`, `action` and `source` are CHECK constraints rather than enums, so adding
one is a migration rather than a type change, and a bad value never reaches a row.

### Who made it

[internal/api/actor.go](../../acceleration/internal/api/actor.go) reads three headers:
`X-Stream-Client` (`dashboard`, `cli`, `sdk`), `X-Stream-Actor-Id` and
`X-Stream-Actor-Name`. They are read **only from a server-side caller**, and a caller that
names no client is recorded as `api`, which is all that can be said about it.

Nothing signs these headers, and nothing needs to. They buy a **name beside a change
somebody already held the credential to make**, never permission to make it: the
credential decides whether the write happens, and the header decides what the row says
about who asked for it. A caller willing to forge a name is a caller who could have made
the change anonymously instead, which is a worse outcome for them, not a better one.

The name is a person's name. The router stores no email addresses, and the dashboard sends
`first_name last_name` rather than the address it has.

### Writing an entry

`Server.audit` in [internal/api/audit.go](../../acceleration/internal/api/audit.go) is
called after every configuration write, in `configs.go`, `config_patch.go`, `sync.go`,
`knowledge.go`, `knowledgeurls.go`, `routerconfigs.go`, `plugins.go` and `policies.go`. It
never returns an error: a change that was made is a change that happened, and failing the
caller's request because the history could not be written would be the wrong trade. A
failure is logged.

What keeps the log readable:

- `auditDiff` marshals both sides and compares field by field, so **a write that moved
  nothing records nothing**. Saving a form without touching it leaves no row.
- `auditBookkeeping` — `id`, `created_at`, `updated_at`, `sync_hash` — is excluded from
  every diff. They move on every write and mean nothing to a reader.
- A plugin's secret is never in the log: `pluginClientOf` carries `{client_id,
  has_secret}`, and a credential removed records `client: "set" → ∅`.
- A policy is diffed against the **stored document**, not the API response, so budget spend
  ticking up is not mistaken for somebody editing the budget.
- A delete reads the row first, so the entry can say what was deleted rather than only
  that something was.

A `synced` entry is the one entry allowed to have no changes. It is not a change: it marks
the moment a directory and the stored config agreed, which is what the conflict check
measures from. It is written **after** the skills and simulations the directory brought
with it — written before them, each of those would look like an edit made since the sync,
and the next sync would refuse itself over its own writes.

### Reading it

- `POST /v1/audit/query` — the log, newest first, filtered by resource type, resource,
  agent, source or action, paged by opaque cursor, 25 a page and 200 at most.
- `GET /v1/agents/configs/{id}/changes` — what changed about one agent **since its
  directory was last synced**: the entries after the newest `synced` entry, plus
  `last_change` and `synced_at`. This is the question the CLI asks.

Both are server-side only.

### The sync that stops

`SyncAgentRequest` gained two fields:

- `check_changes` — refuse the sync, with `409 unsynced_changes`, when it would write over
  a setting somebody changed since the last sync. Omitted, the sync writes over whatever is
  there, which is what a process syncing on startup wants and what every existing caller
  keeps getting.
- `base_change` — the newest change the caller has already seen, as `last_change` named it.
  Sent, the sync is not refused for anything up to and including it. This is how somebody
  says they have looked at what they are about to replace.

The check in [internal/api/sync_conflict.go](../../acceleration/internal/api/sync_conflict.go)
asks a narrower question than "has anything changed". It compares **what the sync would
store** against **what is stored**, field by field, and only for fields the directory
actually declares. Two consequences, both deliberate:

- A setting the directory says nothing about was never at risk, so changing the model in
  the dashboard does not block a sync that only touches the instructions.
- Once the CLI has merged the dashboard's values into the local files, the directory holds
  what the backend holds, there is nothing left to write over, and **the next sync goes
  through with no acknowledgement at all**. The merge path self-heals; `base_change` is
  only needed by somebody deliberately replacing something.

An unknown `base_change` is ignored rather than refused: a client that sends an id the log
does not have is a client that has seen nothing, which is where it started.

### The dashboard

`src/components/dashboard/agents/` in volt-dashboard:

- **Activity** (`/agents/activity/`) — every change to the app, filtered by agent, type or
  client, paged by the same cursor. In the sidebar under the ungrouped block, beside Logs.
- **Recent changes** — the last five changes to one agent, at the bottom of its Behavior
  tab, so somebody about to edit an agent sees that a colleague or a sync got there first,
  with a link to the full history filtered to that agent.

Every dashboard request to the router carries `X-Stream-Client: dashboard` and the
signed-in user as the actor. The local dev proxy forwards them untouched;
`tests/unit/agents/proxy.test.ts` pins that, because a proxy that stripped them would leave
every change attributed to nobody.

### The CLI

`getstream agents sync` now sends `check_changes` by default. On a refusal it reads
`GET /v1/agents/configs/{id}/changes`, prints what changed and who changed it, and asks:

```
jean was changed by Ada Lovelace since you last synced it.
  updated the agent, by Ada Lovelace, Oct 7 10:00
    instructions: "Be brief." → "Be warm."

What should happen to those changes?
> Keep them: write them into my files, then sync
  Replace them with what is in my files
  Stop and leave both alone
```

Keeping them runs `agentrc.Merge`
([internal/agentrc/merge.go](../../../cli/internal/agentrc/merge.go)), which writes each
change into the file it belongs in — `instructions.md`, `guardrail.md`,
`skills/<name>.md`, `knowledge/<source>`, and keys in `agent.yaml` edited through
`yaml.Node` so the comments and the order of the file survive. **Only the files a change
names are rewritten**: this runs on somebody's working copy, and the rest of it is theirs.
The directory is then read again — it is a different directory now, with a different hash
— and synced.

`--merge-changes` and `--overwrite` answer in advance, for CI and for somebody who already
knows. With neither flag and no terminal to ask, the sync **stops**: both other answers
lose somebody's work, and guessing which is not the CLI's call.

A merge that writes nothing — the change was to a plugin credential or a policy, which the
directory keeps no file for — says so and refuses rather than syncing anyway, which would
lose the change it was asked to keep.

## Tests

[internal/api/audit_test.go](../../acceleration/internal/api/audit_test.go) is the log: what
a change records, that a save moving nothing records nothing, that a skill is filed under
its agent and a router under none, that a caller naming no client is `api`, the filters, the
paging, a cursor the list never handed out, another app's changes, and the server-side
posture.

The sync-conflict cases are in
[internal/api/sync_test.go](../../acceleration/internal/api/sync_test.go): the point a sync
marks, the changes since it, the refusal, the field the directory does not declare, the
directory that already holds the change, the acknowledgement, and the sync that did not ask
to be checked and still writes over everything.

The CLI's flow is in `internal/agents_changes_test.go` and the merge in
`internal/agentrc/merge_test.go`, in the cli repository.

## What is not done

- **Nothing outside agents writes to it yet.** The table and the query endpoint are general
  — `resource_type` already names seven kinds — but only the agents surfaces call
  `Server.audit`. Telephony, connectors and SIP trunks are configuration by the same
  measure and should.
- **No retention.** Rows accumulate. A customer with a busy config will want a window, and
  there is no job that trims one.
- **No diff of the text itself.** A changed instruction records the whole before and the
  whole after; the dashboard shows the field name and the CLI shows the first sixty
  characters. Neither shows what moved inside a long prompt.
- **The merge cannot carry everything.** A plugin credential, a policy and a knowledge url
  have no file in the directory, so the merge path refuses rather than dropping them.
- **No way to undo one.** The log says what a field held before; nothing puts it back.
