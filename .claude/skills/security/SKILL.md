---
name: security
description: >-
  Security review of the router: the Go API in acceleration/ (who may reach which
  endpoint, API key and token auth, one user posing as another, tenant isolation,
  SSRF, secrets) and the Kubernetes clusters the hosted router runs on. Use when
  asked for a security review, a security audit, an auth or permissions check, a
  pentest-style pass over a PR or a branch, or whether the cluster is hardened.
---

# Security review

Two reviews, run either or both. Both are read-only: report findings, do not fix them
unless asked.

| Review | Read | Covers |
|---|---|---|
| Go | [go.md](go.md) | `acceleration/`: endpoint access, auth, user impersonation, tenancy, input that reaches the network, the shell or SQL |
| Kubernetes | [k8s.md](k8s.md) | The GKE clusters the hosted router runs on: identity, RBAC, exposure, network, pod security, secrets, supply chain |

Pick from the request: a PR, a diff or "the API" is the Go review; "cluster", "k8s",
"GKE" or "infra" is the Kubernetes one; "security review" with nothing else is both.

For a PR or a branch, review the diff against the rules and also run the coverage checks
in [go.md](go.md#coverage): a new endpoint is a hole if the spec-walking tests do not
reach it.

## Report

One list, sorted by severity, each finding with where it is and what fixes it:

```markdown
# Security review: <what was reviewed>

## CRITICAL
1. **`POST /v1/agents/sessions/query` lists another user's sessions** (go rule 3).
   - `internal/api/session_query.go:39`: `user_id` is honoured for a jwt caller.
   - Fix: narrow to the principal's user unless `ServerSideFrom(ctx)`; add a test in `SessionOwnershipSuite`.

## HIGH / MEDIUM / LOW
...

## Checked and fine
- <rule>: <one line of evidence>
```

CRITICAL is a caller reaching data or actions that are not theirs. HIGH is a guard
missing that nothing else covers. MEDIUM is defence in depth. LOW is hygiene. Say what
you could not check and why rather than leaving it out.
