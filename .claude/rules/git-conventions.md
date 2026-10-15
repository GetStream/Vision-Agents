# Git Conventions

## Branch Naming

```
type/ticket-short-description
```

**Prefixes:**
| Prefix | Use |
|--------|-----|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `hotfix/` | Urgent production fixes |
| `refactor/` | Code restructuring |
| `chore/` | Maintenance, dependencies |
| `docs/` | Documentation only |

**Examples:**
- `feature/proj-1218-csv-export`
- `fix/proj-1028-date-parsing`
- `hotfix/proj-1175-login-redirect`

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description
```

**Types:**
| Type | Use |
|------|-----|
| `feat` | New functionality |
| `fix` | Bug fix |
| `refactor` | Code restructuring (no behavior change) |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance, dependencies, CI |
| `docs` | Documentation only |

**Scope** is optional — use package or area name when helpful (e.g., `api`, `store`, `cli`).

**Examples:**
- `feat(api): add CSV export`
- `fix(store): handle nil pointer in order query`
- `refactor: simplify config loading`
- `chore: upgrade Go dependencies`

**Notes:**
- Ticket numbers belong in the **branch name** and **PR title**, not in every commit.
- Use imperative mood ("add", not "added" or "adds").
- Keep the first line under 72 characters.
