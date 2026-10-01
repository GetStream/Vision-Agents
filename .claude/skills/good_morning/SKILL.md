---
name: good_morning
description: Pull and merge the latest code across the workspace repos (chat, Vision-Agents, artemis-impl, getstream.io, volt-dashboard). Use when the user says good morning or asks to sync their repos.
---

# Good morning

Bring every repo below up to date. Run them in parallel; each is independent.

| Repo | Branch | Then merge |
| --- | --- | --- |
| `~/workspace/chat` | current | `origin/master` |
| `~/workspace/Vision-Agents` | `accelerate` | |
| `~/workspace/artemis-impl` | current | |
| `~/workspace/getstream.io` | current | `origin/main` |
| `~/workspace/volt-dashboard` | `ai-team/agent-dashboard` | |

For each repo:

```bash
cd ~/workspace/<repo>
git fetch origin --prune
git checkout <branch>                         # only when the table names one
git pull --no-rebase --autostash              # skip if the branch has no upstream
git merge --autostash --no-edit origin/<base> # only when the table names one
```

## Rules

- Merge, never rebase, and never force anything.
- Uncommitted work stays put: `--autostash` stashes and reapplies it. If `checkout` refuses
  because of local changes, stop for that repo and report it rather than stashing by hand.
- On a merge conflict, run `git merge --abort` and report the conflicting files. Don't resolve
  them unless the user asks. Once resolved, `git commit` reapplies the autostash itself; never
  `git stash pop` afterwards, which would pop an unrelated older stash.
- If reapplying the autostash conflicts, git keeps it in `git stash list`. Leave the conflicted
  files as they are and report them.
- In `getstream.io`, `content/docs` is now a plain directory of the repo. If an old nested
  checkout of docs-content is still there, the pull refuses to overwrite it: stop and ask, since
  that checkout may hold unmerged work.
- Don't push.

## Report

One line per repo: branch, how many commits came in (`git rev-list --count ORIG_HEAD..HEAD`
or compare `HEAD` before and after), and any conflict or skipped step.
