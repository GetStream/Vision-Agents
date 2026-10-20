---
name: accelerate-sales-engineer
description: "Build a short Stream sales-engineering demo for a customer call: one Accelerate agent, their branding, a streaming web chat UI, and a tunnel link to review. Use this whenever someone asks for a customer demo, a POC, a call demo, a sales-engineering build, a tailored agent for a prospect, or to make something to show a company from a name and URL. Also use it when they mention the Stream CLI, calypso, A2UI, or a review tunnel for a prospect, even if they never say demo or skill."
compatibility: Stream CLI beta (getstream), Slack, Node 22, cloudflared or ngrok
---

# Accelerate sales engineer
Build a small web demo for a customer call. The demo shows one Accelerate agent doing one job in their product, in a chat UI they already recognize. Research their compeititors, pull their website branding, and create a demo that fits to their brand, vertical and use-case.

## Collect the brief

Ask only for what is missing. Company and URL are required. The rest have defaults, used only when the request does not say: "a fast text agent" is text only.

| Input | Default |
| --- | --- |
| Company | required |
| URL | required |
| Modality | Text and voice. Also: text only, voice only, video |
| Top 5 competitors | Run `prospect-competitor-research` |
| Existing customer | Look it up |

Write `BRIEF.md` in the demo repo before coding. It is the contract the last step checks: company, URL, modality, the one use case, five competitors in a line each, SDK and product, and what is mocked.

## Research, then code

Read their site and about page. Use their name, their colors, and the words they use for the product. A recognizable header is enough. Do not clone the marketing site.

Competitors shape the use case. Follow the `prospect-competitor-research` skill with the company and URL, passing along any competitors they named, or reuse its `competitor-research.md` if the company's directory already has one. Take the five competitors from its report, and pick a job from its AI feature tracker that one of them already sells, so the call is about Stream on a workflow the room already believes in.

Existing customer: search Slack for a channel named like the company (`{slug}-internal`, `{slug}-stream`, or a shared channel) and read it for the SDK and the product they already run. Internal channels are often private, so ask before searching private ones. Ask the user only for what the channel does not say. A chat customer gets `stream-chat-react`. Match a video customer the same way when the modality needs a call.

No channel, and nobody says they are a customer: treat them as net-new. Infer the use case from the site and the competitors. Do not invent a relationship.

## How to build it

Put the project in a new directory named for the company, outside any repo and outside the skills folder, and say the path.

Install the beta CLI when `getstream --version` is missing or is semver. A beta build is a date and a hash.

```bash
curl -fsSL https://getstream.io/cli/beta/install.sh | bash
```

If it needs a login, run `getstream login` and send the user the URL immediately, then keep researching while they approve it.

Follow the agents guide rather than this file. Fetch it and do what it says:

```bash
curl -fsSL https://getstream.io/SKILL-agents.md
```

That means `getstream init`, then `getstream agents init` with a template that matches the modality, then `getstream docs` for the platform quickstart. Read `-h` before inventing a flag.


The head of the SDK live on this Github branch: https://github.com/GetStream/Vision-Agents/tree/accelerate. Always pull the latest code and check the API and SDKs before starting to code. The API spec can be found here: https://github.com/GetStream/Vision-Agents/blob/accelerate/acceleration/api/openapi.yaml

The page is web. The agent client is `@stream-io/calypso`. Chat UI is `stream-chat-react`, with `stream-chat-react/ai-components` for the streaming reply. Voice and video go through calypso, not a second client stack.

- **Text and voice** (default): one conversation they can type or talk in.
- **Text only**: chat, no call.
- **Voice only**: the call is the path. Chat may show the transcript.
- **Video**: the call includes video. Still one path.

The thinking state belongs inside the agent's reply bubble, not as a status line under the chat. The router creates the reply as soon as it starts, empty and with `generating` true, then streams text into it. Until the first words arrive, render `AIStateIndicator` from `stream-chat-react/ai-components` in that bubble: the typing dots and a thinking line, worded from the channel's `ai_indicator` state. Then the same bubble streams the answer. A blank pause looks like a frozen demo. Leave thinking on.

A2UI is one interactive surface on that path: a card the user acts on, which the agent treats as the next turn. Render it after the reply text, not under the dots. Mock auth and any system the agent calls. A demo user with a locally minted token is enough. Label mocked data as mock so a caller is not surprised.

Tools the agent calls run in your own worker process (`Dispatch` in calypso). The page still loads without it, but the reply hangs on the tool call, so keep it running and restart it if it exits.

## While it runs

Send a public URL as soon as the page serves, and again when the critical path works. Use `cloudflared tunnel --url http://localhost:<port>`, adding `--protocol http2` if the tunnel never answers, or `ngrok http <port>` when that is what is installed. Do not wait for polish, and do not send a link on every save.

Also write `SCRIPT.md`: the opening line, the three clicks or sentences on the critical path, and what the agent does back. Short enough to read while the call connects. Use their services, not a generic assistant.

## Before you show it

Do not send "it's ready" until this list matches `BRIEF.md`.

- Their name, URL, and branding are on the page.
- Modality is the one they asked for.
- Five competitors are in the brief, and the use case comes from them or from the customer's real product.
- Slack was searched. The SDK and product match the channel, or the brief says net-new and the use case is inferred.
- The critical path works through the tunnel link: the agent answers, its reply bubble shows the typing dots and thinking state before the text streams in, the A2UI surface does one real action, auth is mocked.
- The page is web, on calypso, with the React Chat UI for a chat customer.

Then run `/simplify` on the demo and apply the small fixes. If that command is not available, do the same pass yourself: drop comments that restate the code, one-off helpers, and anything the call does not exercise. Re-check the list after edits. Then send the tunnel link and the script.
