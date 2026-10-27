---
name: docs
description: How to edit the agents docs on getstream.io. Use when writing or updating agents documentation pages, sidebars or the SDK switcher.
---

Docs live in the main site repo, typically located at workspace/getstream.io, under `content/docs`

Some general docs tips

- Keep it short
- Show examples before explaining details
- Write from the code, not from memory. Check the SDK (`sdks/<lang>`) and `acceleration/api/openapi.yaml` before documenting an API, key or default

## How a page reads

Write it so it needs no deslop pass. `agents/_default/02-agents/creating-an-agent.md` is the reference page, and Thierry's deslop commits in getstream.io (7ee27d28eb, 21c69fd256, edf81f51f2, 1d93015abe) show the cuts.

- Open with why the reader wants it and one real example, not a definition: "Your support agent shouldn't answer: Please solve the Navier-Stokes problem", not "A guardrail screens what a caller asks"
- Example first: the `agent.yaml` or the call, then the command to try it (`getstream agents test support --sync`) and a question to ask. Then explain only what the example doesn't show
- Real example values (`dentist_ai`, `acme-workspace`, "What is your return policy?"), never `foo` or `my-thing`
- One or two sentences per paragraph, written like a person: "It returns the connection", not "The answer is the connection". Contractions are fine
- Where there are options, recommend one and say when to pick the other: "We recommend each user's own account for personal data, and a shared account for company tools"
- One H2 per thing the reader wants to do. A fact lives on one page and other pages link it. End with `## Going further` and `<Cards>`

Leave out:

- Validation rules, limits and error codes ("refused with a 400", length caps, "refused rather than ignored"). A troubleshooting page may list the errors people really hit, by their message
- Internal mechanics the reader never touches: channel names, sync fingerprints, our staging clusters and test apps
- Raw HTTP paths when an SDK or the CLI does it. A call with no SDK method gets one curl block
- Client setup boilerplate in snippets: show the call
- Sentences that restate the code above them, "To recap" tables that restate the page, and signposting ("The next section explains")
- Which SDK supports what "for now". When only some SDKs have a feature, one `<Admonition>` names them
- Bold lead-ins in bullets or prose (bold is for step titles), and hype ("amazed", "effortlessly")

## Only what works

- A page says only what a user can install and run today, and you saw work end to end: run every snippet, curl and command, against staging if needed. Reading the code confirms a name, not a behaviour
- "Which SDKs support X" means released packages (npm, PyPI, Maven Central, pub.dev, the released CLI), not source on `accelerate`
- Not proven yet? Leave it out and ticket it. Don't caveat it ("not tested yet")

## Where things live

- Content: `getstream.io/content/docs`, a plain directory of the getstream.io repo (it was GetStream/docs-content). Its authoring conventions are in `content/docs/AGENTS.md`
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
- Commit content on a getstream.io branch, never `main`, and open an ordinary PR. It deploys when merged
