---
name: docs
description: How to edit the agents docs on getstream.io. Use when writing or updating agents documentation pages, sidebars or the SDK switcher.
---

Docs live in the main site repo (GetStream/getstream.io) under `content/docs`, typically located at workspace/getstream.io

Some general docs tips

- Keep it short
- Show examples before explaining details
- Write from the code, not from memory. Check the SDK (`sdks/<lang>`) and `acceleration/api/openapi.yaml` before documenting an API, key or default

## Where things live

- Content: `getstream.io/content/docs`, a plain folder in the site repo (no longer a submodule)
- Pages: `agents/<framework>/*.md`; pages shared by every SDK go in `agents/_default/`
- Sidebars: `_sidebars/[agents][<framework>].json` (`baseURL`, `defaultLanguages`, `items` of `{ title, slug, markdown }`). `[agents][default].json` is the `/agents/docs/` home
- Framework slugs: `javascript`, `ios` (shown as Swift), `android` (Kotlin), `flutter` (Dart), `python`, `go-golang`, `dotnet-csharp`, `ruby`, `rust`, `php`. A new slug needs a label in `routing.ts` and a logo in `public/docs-assets/icons/`
- SDK switcher and product menu: `getstream.io/src/docs/utils/routing.ts` and `productMenu.ts`. A new sidebar must be reachable from them

## Writing

- The page H1 comes from the sidebar `title`, so don't put a `#` heading in the markdown
- Use `$framework` in links (`/agents/docs/$framework/quickstart/`) so shared pages work under every SDK
- Components: `<Cards>`, `<Card icon title description href>`, `<Admonition type="info">`, `<Tabs>`, `<Steps>`, mermaid fences. `icon` takes Remixicon names (an unknown name fails the build); `logo="go"` takes SDK logos
- Only ASCII, and no em dashes. Links are root-relative, and slugs are lowercase with hyphens

## Gotchas

- Every server SDK reads `agent.yaml` and syncs the directory in one request, skipping it when `.agent_sync` matches. They differ in when: Python syncs automatically on starting, the others when you call `sync`. Say which SDK a behaviour applies to
- Client SDKs (JavaScript in the browser, Swift, Android, Flutter) never sync or configure agents. Swift and browser JavaScript are handed a config id by the backend; Android and Flutter open an agent by its stored name
- `SyncRouters` rejects unknown keys in `routers/*.yaml` (including `description`)

## Preview and ship

- `npm run dev` in getstream.io (Node 24), then open `http://localhost:3000/agents/docs/<framework>/`. Don't run builds or checks unless asked
- Commit content and site changes together on a getstream.io branch, never `main`
