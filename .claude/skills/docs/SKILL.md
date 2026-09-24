---
name: docs
description: How to edit the agents docs on getstream.io. Use when writing or updating agents documentation pages, sidebars or the SDK switcher.
---

Docs are a submodule on the main site, typically located at workspace/getstream.io

Some general docs tips

- Keep it short
- Show examples before explaining details
- Write from the code, not from memory. Check the SDK (`sdks/<lang>`) and `acceleration/api/openapi.yaml` before documenting an API, key or default

## Where things live

- Content: `getstream.io/content/docs` (repo GetStream/docs-content). Agents work happens on its `agents` branch
- Pages: `agents/<framework>/*.md`; pages shared by every SDK go in `agents/_default/`
- Sidebars: `_sidebars/[agents][<framework>].json` (`baseURL`, `defaultLanguages`, `items` of `{ title, slug, markdown }`). `[agents][default].json` is the `/agents/docs/` home
- Framework slugs: `javascript`, `ios` (shown as Swift), `python`, `go-golang`
- SDK switcher and product menu: `getstream.io/src/docs/utils/routing.ts` and `productMenu.ts`. A new sidebar must be reachable from them

## Writing

- The page H1 comes from the sidebar `title`, so don't put a `#` heading in the markdown
- Use `$framework` in links (`/agents/docs/$framework/quickstart/`) so shared pages work under every SDK
- Components: `<Cards>`, `<Card icon title description href>`, `<Admonition type="info">`, `<Tabs>`, `<Steps>`, mermaid fences. `icon` takes Remixicon names (an unknown name fails the build); `logo="go"` takes SDK logos
- Only ASCII, and no em dashes. Links are root-relative, and slugs are lowercase with hyphens

## Gotchas

- Go and Python differ: Python reads `agent.yaml` and syncs it automatically, but Go builds the config in code and calls `Sync`. Say which SDK a behaviour applies to
- `SyncRouters` rejects unknown keys in `routers/*.yaml` (including `description`)

## Preview and ship

- `npm run dev` in getstream.io (Node 24), then open `http://localhost:3000/agents/docs/<framework>/`. Don't run builds or checks unless asked
- Commit content on the submodule branch and push it, then commit the submodule pointer bump on a site branch, never `main`. Merge docs-content first
