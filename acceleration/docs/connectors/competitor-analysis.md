# Voice-agent connectors: competitor analysis

Sep 28, 2026 · @Kanat Kiialbaev

Exported from Claude Docs on 2026-10-01 (https://claude.ai/code/artifact/aaab7491-0e30-469e-81cb-8848fd2218af). The Claude Doc is the source of truth; this copy is a snapshot.

## Contents

1. [Why this document](#mgrnb5r1kqa.41733)
2. [TL;DR](#mgrnb5r1kqa.40003)
3. [What MCP is](#mgrnb5r1kqa.175730)
   - [What problem it solves](#mgrnb5r1kqa.176153)
   - [Roles: host, client and server](#mgrnb5r1kqa.176745)
   - [What a server offers: tools, resources, prompts](#mgrnb5r1kqa.178382)
   - [Messages and transports](#mgrnb5r1kqa.179702)
   - [How one tool call works](#mgrnb5r1kqa.182299)
   - [MCP in this document](#mgrnb5r1kqa.183835)
   - [MCP terms](#mgrnb5r1kqa.187146)
4. [Context: what Router and Athena are](#mgrnb5r1kqa.131520)
   - [Router](#mgrnb5r1kqa.132161)
   - [Athena](#mgrnb5r1kqa.136889)
   - [Who calls whom](#mgrnb5r1kqa.140007)
   - [How plugins work today](#mgrnb5r1kqa.198696)
   - [Connectors and how they differ from plugins](#mgrnb5r1kqa.206222)
   - [Where connectors fit](#mgrnb5r1kqa.140651)
5. [The middle layer: what it is and do we need it](#mgrnb5r1kqa.157106)
   - [What MCP does and does not do](#mgrnb5r1kqa.157445)
   - [How Vercel does it: Eve and Connect](#mgrnb5r1kqa.158503)
   - [What matches it in AI-816](#mgrnb5r1kqa.160832)
   - [Answer to Thierry's question](#mgrnb5r1kqa.161572)
6. [Glossary](#mgrnb5r1kqa.46471)
7. [Scope and method](#mgrnb5r1kqa.63)
   - [Evaluation criteria](#mgrnb5r1kqa.162092)
8. [LiveKit](#mgrnb5r1kqa.12243)
   - [What LiveKit Connectors are](#mgrnb5r1kqa.12528)
   - [Channels and tools](#mgrnb5r1kqa.13644)
   - [MCP](#mgrnb5r1kqa.14190)
   - [Credentials](#mgrnb5r1kqa.15024)
   - [Call runtime](#mgrnb5r1kqa.15424)
   - [Assessment (our view)](#mgrnb5r1kqa.16114)
9. [Pipecat](#mgrnb5r1kqa.2806)
   - [Integrations: channels and tools](#mgrnb5r1kqa.2994)
   - [MCP](#mgrnb5r1kqa.3721)
   - [Credentials](#mgrnb5r1kqa.4597)
   - [WhatsApp and Twilio](#mgrnb5r1kqa.5057)
   - [Call runtime](#mgrnb5r1kqa.5578)
   - [Assessment (our view)](#mgrnb5r1kqa.6253)
10. [Retell AI](#mgrnb5r1kqa.6988)
    - [Catalog: marketing vs reality](#mgrnb5r1kqa.7299)
    - [The App model (connection)](#mgrnb5r1kqa.8045)
    - [How providers authenticate](#mgrnb5r1kqa.9016)
    - [Integration tools, custom functions, MCP](#mgrnb5r1kqa.9699)
    - [Assessment (our view)](#mgrnb5r1kqa.11259)
11. [ElevenLabs Agents](#mgrnb5r1kqa.16971)
    - [Tools, integrations, channels](#mgrnb5r1kqa.17270)
    - [Webhook tools: one envelope for all tools](#mgrnb5r1kqa.18307)
    - [Auth connections](#mgrnb5r1kqa.19127)
    - [MCP](#mgrnb5r1kqa.20295)
    - [Runtime and observability](#mgrnb5r1kqa.21144)
    - [Assessment (our view)](#mgrnb5r1kqa.21733)
12. [PolyAI](#mgrnb5r1kqa.22599)
    - [Integrations and channels](#mgrnb5r1kqa.22951)
    - [Four ways to build an integration](#mgrnb5r1kqa.23595)
    - [Credentials](#mgrnb5r1kqa.24676)
    - [Contact centers and handoff](#mgrnb5r1kqa.25657)
    - [Runtime](#mgrnb5r1kqa.26056)
    - [Assessment (our view)](#mgrnb5r1kqa.26476)
13. [Comparison](#mgrnb5r1kqa.27281)
    - [Facts](#mgrnb5r1kqa.27554)
    - [Assessments (our view)](#mgrnb5r1kqa.29050)
    - [Why competitors built it this way](#mgrnb5r1kqa.162108)
14. [Other products](#mgrnb5r1kqa.80042)
    - [Other voice and CX platforms](#mgrnb5r1kqa.88753)
    - [Enterprise platforms: AWS, Salesforce, Microsoft, Google](#mgrnb5r1kqa.81434)
    - [Assistants and IDEs](#mgrnb5r1kqa.84596)
    - [Brokers and iPaaS](#mgrnb5r1kqa.91761)
15. [Build our own layer or use a broker](#mgrnb5r1kqa.49663)
16. [Whose OAuth app: customer lock-in, brand and risks](#mgrnb5r1kqa.103491)
    - [Where lock-in comes from](#mgrnb5r1kqa.104970)
    - [What AI-816 already has](#mgrnb5r1kqa.105875)
    - [What others do](#mgrnb5r1kqa.106693)
    - [What it costs us if the app is ours](#mgrnb5r1kqa.108183)
    - [What it costs us if the app is the customer's (our view)](#mgrnb5r1kqa.109638)
    - [Where the lock-in really is (our view)](#mgrnb5r1kqa.109982)
    - [Conclusion (recommendation)](#mgrnb5r1kqa.110759)
17. [Personal token in a voice channel](#mgrnb5r1kqa.75938)
18. [Edge cases](#mgrnb5r1kqa.30446)
    - [For Athena](#mgrnb5r1kqa.55357)
    - [Incidents and provider quirks](#mgrnb5r1kqa.95263)
19. [iMessage](#mgrnb5r1kqa.191573)[: a channel, not a connector](#mgrnb5r1kqa.191573)
    - [Why iMessage is a channel, not a tool](#mgrnb5r1kqa.192024)
    - [Two paths](#mgrnb5r1kqa.192742)
    - [Linq: why they are big](#mgrnb5r1kqa.396131)
    - [Voice agents and Router](#mgrnb5r1kqa.194707)
    - [Conclusion (recommendation)](#mgrnb5r1kqa.195669)
20. [What this means for Accelerate](#mgrnb5r1kqa.33382)
    - [Keep: strong parts of the design](#mgrnb5r1kqa.34304)
    - [Work plan](#mgrnb5r1kqa.35017)
    - [Decide separately](#mgrnb5r1kqa.36717)
    - [Risks of our approach (assessment)](#mgrnb5r1kqa.69230)
21. [Open questions](#mgrnb5r1kqa.37294)
    - [Product questions](#mgrnb5r1kqa.37311)
    - [Need people or accounts](#mgrnb5r1kqa.174466)
22. [Appendix: fact check (September 29)](#mgrnb5r1kqa.162145)
23. [Sources](#mgrnb5r1kqa.38890)

## Why this document

Accelerate is Stream's platform for voice and text agents: a Go router in `acceleration/` of the Vision-Agents repo. Ticket [AI-816](https://linear.app/stream/issue/AI-816/basic-connectorsmcp-support) (branch [`codex/connector-support`](https://github.com/GetStream/Vision-Agents/tree/codex/connector-support)) has a prototype of connectors that would replace today's plugins. It is an example from Nash, not an accepted design, and it can be redone (see «Connectors and how they differ from plugins»). With connectors, the agent calls tools of other services (Slack, Linear, Salesforce…) over [MCP](#mgrnb5r1kqa.175730), and Accelerate stores and refreshes the OAuth tokens.

- **First user**: Athena, Stream's internal AI assistant. Example task: «read a Slack thread and summarize it».
- **Volt**: the dashboard where agents and connections are set up (separate repo `volt-dashboard`).
- **Team question.** /». Neevash Ramdial (Nash), who wrote the first version of AI-816, [listed the gaps](https://getstream.slack.com/archives/C094V4M57NE/p1790193515667049): Google Suite, Slack, Linear, WhatsApp and «MCP layer needs a review (for auth as well)».

**The decision is ours.** In a [thread](https://getstream.slack.com/archives/C094V4M57NE/p1790702241273369) on September 29, asked «which scenario do we support first», Thierry answered: «i think its best if you deeply understand this topic and decide» and «if i have to go and learn the details about this, it takes as much time as building this, we just can't scale like that».

- **There is no clear requirement.** Nobody has said exactly how connectors should work.
- **There is only a frame:** «people using our framework should be able to integrate slack, cal, salesforce etc according to how they want to integrate it».
- **So we must learn the topic in depth and decide ourselves.** This document is the basis for that decision, not a request for one.

The document compares how five voice platforms and other products (enterprise agent platforms, assistants, auth brokers) solve this problem, and proposes what to keep, add or buy.

**What we need from the reviewer:**

1. Check the decision «we support both scenarios, and the developer picks the owner for each integration», and the work order: app-owned first, then personal for Slack and Linear.
2. A decision on the auth broker: build it, buy it, or a mix.
3. Review the work plan in «What this means for Accelerate».

**Reading order.** TL;DR → What MCP is (if the protocol is new to you) → Context → The middle layer → Comparison → Build our own layer or use a broker → What this means for Accelerate → Open questions. «iMessage: a channel, not a connector» answers a separate question about channels and can be read on its own. Use the Glossary as you read. The vendor sections and «Other products» are reference. «Appendix: fact check» is the source of truth if the text disagrees with it. To check the Accelerate section, first read `acceleration/docs/connector-handover.md`, then `connector-design.md` (both on the branch above). GitHub, Linear and Slack links need access to the GetStream org.

## TL;DR

**Do we need a middle layer.** MCP is enough for the tool call itself: all five voice platforms have an MCP client. MCP is not enough to store and refresh tokens and to pick the account: a middle layer does that. We do not need a separate service like Vercel Connect: AI-816 already builds the same layer inside Router. Details: [The middle layer: what it is and do we need it](#mgrnb5r1kqa.157106).

**Decision.** We support both scenarios, and the developer picks the owner of each integration (the decision is ours, see «Why this document»). It is one system in Router: a connection has an owner, `app` or `user`, and a binding has a mode, `fixed` or `session`. The credential resolver boundary (`connector-design.md:409`) lets us add a broker for some providers later. The work order keeps us from investing in personal connections too early:

1. **App-owned first (agents for businesses).** Do what competitors do: app-owned connections, static credentials and client credentials, plus calls to the customer's backend. This covers the market competitors serve and needs almost no provider-specific code. Client credentials is not built yet.
2. **Athena.** Its first scenario is Slack with several people in one session: Nash asks for «slack connection and multiplayer» (October 1). Such a session uses only app-owned connections, so Athena starts with an app-owned Slack connection on Stream's internal app. Personal connections on Slack and Linear come after that, with a live test. Do not grow the catalog until we decide whether we need a broker (see «Build our own layer or use a broker»). Decide on the broker from data: how much work two providers take over a month of running.

&#91;embedded content: one layer for two scenarios · AI-816 model\]

Both scenarios go through the same AI-816 entities. Only two fields differ: the connection owner (`app` or `user`) and the binding mode (`fixed` or `session`). The provider examples for a business agent come from Thierry's words «slack, cal, salesforce». The broker is a possible future option, not a plan (`connector-design.md:409`).

**Main findings.**

- **The credential owner depends on the channel, not the vendor.** Where the user has a browser or an IDE, personal OAuth is the norm. In voice and contact centers it is a service account, even at Microsoft, Google and Salesforce ([Other products](#mgrnb5r1kqa.80042)).
- **The five voice platforms keep credentials at the workspace level or do not store them at all.** That makes sense for their market: a company's agent talks to its customer on the phone and works with the company's systems on the company's behalf. We need OAuth on behalf of a user mainly for Athena, an employee assistant like Claude or ChatGPT connectors ([Why competitors built it this way](#mgrnb5r1kqa.162108)).
- **Our decision matches AWS AgentCore, Salesforce, Copilot Studio and Google ADK.** AWS has a third mode: token exchange without consent. OpenAI and Glean have a separate account for the agent.
- **In voice, a personal token exists only if the account was linked before the call or the token comes from an exchange.** A link sent during a call must be bound to the session. A device code read out by voice is a known phishing vector ([Personal token in a voice channel](#mgrnb5r1kqa.75938)).
- **The AI-816 data model is good, but the main part is not proven.** Connection, owner and binding are the same scheme as App at Retell and auth connections at ElevenLabs, plus a user as owner. But only 2 of 7 providers were tested live, live refresh was not tested for any, there is no UI for personal connections, latency was not measured, and remaining work has 9 items (`connector-handover.md`).
- **Main edge cases.** A shared agent in a group chat answers one person with another person's token. The refresh response gets lost during rotation. An OAuth client has a grant limit: Google allows 100 refresh tokens per account, Salesforce 5 approvals per user ([Edge cases](#mgrnb5r1kqa.30446)).

&#91;embedded content: who runs auth and whose account it is · 5 competitors and Accelerate\]

The position on the axes is our view, based on the «Facts» table. For contact centers the bottom-right quadrant is the expected place, not a weakness. The top-right is held by brokers and products like Claude and ChatGPT connectors; they are not shown on the chart. Accelerate is placed there by design; this is not proven in production.

**iMessage** (Thierry's question from September 30). Talking to an agent in iMessage is a channel, not a tool, so it is not part of AI-816. Apple has no public iMessage API. The official path is Apple Messages for Business through an Apple-approved provider. Unofficial APIs (Sendblue, Linq) work only on the customer's account. Details: [iMessage: a channel, not a connector](#mgrnb5r1kqa.191573).

**Top priorities.** The full list with reasons is in [Work plan](#mgrnb5r1kqa.35017).

1. A rule for a shared Slack channel: a session with several people uses only app-owned connections.
2. Reliable refresh and a live test of it on Slack and Linear: a Linear access token lives 24 hours.
3. The Slack OAuth app choice: for Athena, Stream's internal app; for the first external customers, their own internal app (BYO), because Slack MCP is not allowed for Stream's app until it is listed in the Marketplace.
4. A phone session uses only app-owned connections by default. Linking personal accounts (a «connect accounts» page, a one-time link bound to the session and the user) moves to P1: Athena starts with sessions of several people.

The risks of this approach are in [Risks of our approach](#mgrnb5r1kqa.69230).

## What MCP is

MCP (Model Context Protocol) is an open standard for connecting an AI app to outside systems: files, databases, service APIs. In this document, Router uses MCP to call tools of Slack, Linear and other services. This section explains MCP from scratch. All facts were checked against [modelcontextprotocol.io](https://modelcontextprotocol.io) on September 29, 2026. The current spec revision is 2026-07-28: [/specification/latest](https://modelcontextprotocol.io/specification/latest) points to it.

### What problem it solves

Without a shared standard, every AI app writes its own integration with every service. Claude Code, VS Code and our Router would each write their own Linear integration, and the same for every other service. With MCP, a service builds an MCP server once, an app builds an MCP client once, and they work together right away. The MCP docs use a plug analogy: «Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems» ([intro](https://modelcontextprotocol.io/docs/getting-started/intro)).

### Roles: host, client and server

MCP has three parties ([architecture](https://modelcontextprotocol.io/docs/learn/architecture)).

- **Host**: the AI app a person works with: Claude Code, VS Code, Claude Desktop. «The AI application that coordinates and manages one or multiple MCP clients». The host talks to the model and decides which tools to show it.
- **Client**: an object inside the host that keeps a connection to one server. The host makes one client per server: «creating one MCP client for each MCP server. Each MCP client maintains a dedicated connection with its corresponding MCP server». Connect two servers to VS Code, and VS Code has two clients inside.
- **Server**: a program that gives the client tools and data: «A program that provides context to MCP clients». There are two kinds:
  - **Local.** For example, the [filesystem server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem), which reads files on disk. It runs on the same machine and usually serves one client.
  - **Remote.** For example, the [Sentry MCP server](https://docs.sentry.io/product/sentry-mcp/), which runs on Sentry's servers and serves many clients at once.

**Who starts whom.**

1. A person starts the host.
2. The host makes one client for each server in its config.
3. The client itself starts a local server as a child process: «the client launches the MCP server as a subprocess» ([stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)).
4. A remote server is already running at the service owner. The client only sends it HTTP requests.

**The model is not part of MCP.** «MCP focuses solely on the protocol for context exchange—it does not dictate how AI applications use LLMs» ([architecture](https://modelcontextprotocol.io/docs/learn/architecture)). The model never talks to a server itself. The host shows it the tool descriptions, and when the model asks to call a tool, the client makes the call.

&#91;embedded content: MCP architecture · host, clients, servers, outside systems\]

The host keeps one client per server. The client starts a local server itself and talks to it over stdin and stdout; it reaches a remote server over HTTP. The model talks only to the host, and not over MCP. The filesystem and Sentry example comes from [architecture](https://modelcontextprotocol.io/docs/learn/architecture).

### What a server offers: tools, resources, prompts

A server can offer three kinds of things, called primitives. They differ in who decides when to use them ([server concepts](https://modelcontextprotocol.io/docs/learn/server-concepts)).

| Primitive | What it is | Who decides when to use it | Methods | Example from the spec |
| --- | --- | --- | --- | --- |
| [Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) | A function the model can call: write to a database, call an API, change a file | The model («model-controlled»). Still, «there SHOULD always be a human in the loop with the ability to deny tool invocations» | `tools/list`, `tools/call` | `get_weather` with the argument `location` returns text about the weather |
| [Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources) | Read-only data at an address (URI): a file, a DB schema, a document | The host app («application-driven»): it decides what to put in the model's context | `resources/list`, `resources/read` | `resources/read` with `file:///project/src/main.rs` returns the file text |
| [Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts) | A ready-made instruction template for the model | The user («user-controlled»), for example through a slash command | `prompts/list`, `prompts/get` | `code_review` returns a list of messages for the model |

It also works the other way: a server can ask the client to check something with the user. This is **elicitation**: a form or a link, and sensitive data is asked for only by link. Two other such features, sampling and roots, are «deprecated as of protocol version 2026-07-28» ([client concepts](https://modelcontextprotocol.io/docs/learn/client-concepts)).

### Messages and transports

**Message format.** Every message is JSON in the JSON-RPC 2.0 format ([basic](https://modelcontextprotocol.io/specification/2026-07-28/basic)).

- A request has `jsonrpc: "2.0"`, an `id` number, a method name `method` (for example, `tools/call`) and arguments `params`.
- A response has the same `id` and either `result` or `error`.

**In revision 2026-07-28 the protocol became stateless** ([changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)):

- **Before** (2025-11-25 and earlier). A connection started with a handshake: an `initialize` request, the server's response and a `notifications/initialized` notification.
- **Now.** There is no handshake. Each request carries the protocol version and the client's capabilities in the `_meta` field. To learn what a server can do, send `server/discover`.

**A transport** is how these JSON messages physically get from client to server. There are two ([transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)).

| Transport | How the bytes travel | When to use it | Auth |
| --- | --- | --- | --- |
| stdio | The client starts the server as a child process. It writes requests to the server's stdin and reads responses from stdout. One message is one line of JSON. The server writes logs to stderr, because stdout is only for MCP messages ([stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)) | The server is on the same machine as the host: files, git, local tools | Not covered by the auth spec: the server takes keys from environment variables |
| Streamable HTTP | Each message is an HTTP `POST` to one address, for example `https://mcp.linear.app/mcp`. The response is one JSON or an SSE stream of events for that request. In 2026-07-28 every request carries the Mcp-Method header (the method name), and tools/call, resources/read and prompts/get also carry Mcp-Name (the tool name or URI) ([Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)) | A remote server that many clients use. The Slack and Linear servers work this way | OAuth 2.1: the token goes in the `Authorization: Bearer …` header |

History you will see in the competitor sections:

- The old HTTP+SSE transport from version 2024-11-05 «has been deprecated since protocol version 2025-03-26». Streamable HTTP replaced it.
- The `Mcp-Session-Id` session header existed in versions 2025-03-26 to 2025-11-25 and was removed in 2026-07-28.

**Auth for HTTP** ([authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)):

- The MCP server is an «OAuth 2.1 resource server»: it accepts requests with a token. The client is an OAuth client: it gets the token.
- The client learns where to get a token from the server's metadata (RFC 9728). The address `/.well-known/oauth-protected-resource` serves it, or a link in the 401 response.
- The token is issued for this exact server (RFC 8707). The server does not pass it on to the service API: «The MCP server MUST NOT pass through the token it received from the MCP client». For calls to its own API the server uses a separate token ([security considerations](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/security-considerations)).
- What the spec does not cover (storing and refreshing the token, picking the account) is covered in «What MCP does and does not do».

### How one tool call works

The example is from the spec ([tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)): a user asks about the weather in New York, and the server has a `get_weather` tool.

1. **Tool list.** Before the conversation, the client sends `tools/list` to the server. The response has a name, a description and a JSON Schema of the arguments for each tool. The host merges the lists from all servers into one.
2. **User request.** The person writes to the host: «What's the weather in New York?»
3. **Model call.** The host sends the model the conversation history and the tool descriptions from step 1. This is a request to the model provider's API, not MCP.
4. **Model decision.** The model answers with a call, not text: the name `get_weather` and the arguments `{"location": "New York"}`.
5. **`tools/call`.** The host catches this call and hands it to the client of the right server. The client sends (the `_meta` field is left out, as in the spec): `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_weather","arguments":{"location":"New York"}}}`
6. **Server work.** The server runs the tool. For a service's server this is usually a request to the service's own API.
7. **Result.** The server answers: `{"jsonrpc":"2.0","id":2,"result":{"resultType":"complete","content":[{"type":"text","text":"Current weather in New York: …"}],"isError":false}}`
8. **Answer to the user.** The host adds `content` to the conversation and calls the model again. Now the model writes the answer text, and the host shows it to the person.

If the tool fails, the server returns `isError: true` with the error text. This text also goes to the model: «Clients SHOULD provide tool execution errors to language models to enable self-correction».

&#91;embedded content: one tool call · sequence diagram, 8 steps\]

Only messages between client and server go over MCP: `tools/list`, `tools/call` and their responses. The conversation with the model and the server's request to its own API do not use MCP. The diagram follows the sequence from [tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools).

### MCP in this document

In AI-816, Router is both the host and the client, and the servers are the services' own remote MCP servers. We do not write our own MCP server. Code: branch `codex/connector-support` @ `7855a261`, checked on September 29.

| MCP role | Who it is in AI-816 | Where in the code |
| --- | --- | --- |
| Host | Session in Router. When a session is created, it connects connectors, gives their tools to the LLM and runs the LLM's calls | `attachConnectors` in `internal/session/manager.go:266` |
| Client | One per session binding. Made by the official Go SDK `github.com/modelcontextprotocol/go-sdk` v1.8.0 | `mcp.Open` → `dial`: `mcp.NewClient` and `client.Connect` (`internal/mcp/mcp.go:197-203`) |
| Server | Remote servers from the catalog: `https://mcp.slack.com/mcp`, `https://mcp.linear.app/mcp`, `https://api.githubcopilot.com/mcp/`, `https://mcp.cal.com/mcp` and others. Slack, Linear and GitHub write and run them | `internal/mcp/connectors.yaml`, field `url` |
| Outside system | The Slack and Linear APIs. The service's MCP server calls them, not Router | — |
| Model (not an MCP role) | The LLM picked by Model router. It sees a connector tool as `<binding>__<tool>`: binding `linear` and tool `create_issue` give `linear__create_issue`. «(via linear)» is added to the description | `Prefix` (`mcp.go:414`), `harness.Tool` (`mcp.go:153-156`) |

&#91;embedded content: MCP in AI-816 · Router, Slack and Linear MCP servers, their APIs\]

Session makes one MCP client per binding. Each client puts the token from its connection into its HTTP requests. Then the service's MCP server calls its own API. The model sees only the tool list and the results; it does not see tokens or server addresses.

**Which part of MCP is used.**

- **Only Streamable HTTP.** The client is made with `StreamableClientTransport` (`mcp.go:198`). There is no stdio: a search for `CommandTransport` and `StdioTransport` in `acceleration/` finds nothing. This makes sense: Router is a cloud service and has no need to run other people's processes (our view).
- **Only tools.** Router calls `tools/list` with all pages by cursor (`mcp.go:215`) and `tools/call` (`mcp.go:300`). Resources and prompts are not used: `acceleration/` has no calls to `ListResources`, `ReadResource`, `ListPrompts`, `GetPrompt`.
- **The tool list is fetched once, when the session starts.** Connecting and `tools/list` get 10 seconds (`startupTimeout`, `mcp.go:25`). The separate SSE stream for server notifications is turned off (`DisableStandaloneSSE: true`, `mcp.go:201`). So a `notifications/tools/list_changed` notification does not reach a running session.
- **Protocol version.** SDK v1.8.0 first sends `server/discover` from 2026-07-28. If the server does not understand it, the SDK falls back to the old `initialize` handshake of version 2025-11-25 (`mcp/client.go:319-404` in go-sdk v1.8.0). Which version the live servers answer with is only partly known (checked October 1). Linear's docs say its server «follows the authenticated remote MCP spec» revision 2025-03-26, with SSE as a deprecated fallback ([Linear MCP](https://linear.app/docs/mcp)). Slack's docs name no revision, only «JSON-RPC 2.0 over Streamable HTTP» ([Slack MCP](https://docs.slack.dev/ai/slack-mcp-server/)). An unauthenticated `server/discover` to both returns 401 with only the `WWW-Authenticate` resource metadata, so the real handshake needs an authorized token (`unverified` for the live answer).

**What Router does on top of MCP before each `tools/call`** (`Runtime.Call`, `mcp.go:272-331`):

1. Checks the model's arguments against the tool's JSON Schema (`mcp.go:286`).
2. Checks that the binding allows the tool and its `schema_digest` has not changed (`mcp.go:296`).
3. Puts the connection token in the `Authorization: Bearer …` header. `authorizedTransport.RoundTrip` (`mcp.go:379`) does this by calling `connectors.AuthorizeRequest`.
4. Limits the call time: 5 seconds by default (`defaultConnectorToolTimeout`, `internal/session/connector_tools.go:19`).
5. Takes only text from the response, up to 32 KB (`maxMCPToolResultBytes`, `mcp.go:27`). Images and other non-text content are replaced with the phrase «MCP tool returned non-text content that the agent cannot display» (`mcp.go:319`).

MCP does not say where the token comes from, who refreshes it, or which account is picked. That is the job of the middle layer, covered in «The middle layer: what it is and do we need it».

### MCP terms

| Term | What it is |
| --- | --- |
| Host | The AI app that talks to the model and holds MCP clients: Claude Code, VS Code; for us, a Router Session |
| MCP client | An object inside the host; a connection to one server |
| MCP server | A program that offers tools, resources and prompts: local or remote |
| Tool | A server function that the model decides to call |
| Resource | Read-only data at a URI; the app decides when to use it |
| Prompt | An instruction template; the user decides when to use it |
| Elicitation | A server request to the user through the client: a form or a link |
| JSON-RPC 2.0 | The message format: JSON with `method`, `params`, `id`, and a response with `result` or `error` |
| `tools/list`, `tools/call` | The methods «give me the tool list» and «call the tool with these arguments» |
| `server/discover` | The request «what can you do and which versions do you support» in 2026-07-28; before, `initialize` did this |
| Transport | How messages are delivered: stdio or Streamable HTTP |
| stdio | The server is a child process of the client; messages are JSON lines on stdin and stdout |
| Streamable HTTP | Each message is an HTTP `POST` to one server address |
| SSE | Server-Sent Events: an HTTP response where the server writes events as they are ready, without closing the connection |
| Token passthrough | Passing the client's token on to the service API. MCP forbids it ([security best practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)) |

Auth terms (PKCE, DCR, CIMD) are in the main «Glossary».

## Context: what Router and Athena are

This section is for readers who see these systems for the first time. In short:

- **Router** is our HTTP service in Go. An app sends it a request, for example «transcribe this recording» or «join this call as an agent». Router calls the API of the right provider (Deepgram, ElevenLabs, OpenAI and so on) and returns the result.
- **Athena** is an internal chat assistant for Stream employees. It has its own server: Athena API in Go with its own database. For model answers this server calls the Router HTTP API.
- **Connectors (AI-816)** is code inside Router. It lets the agent call tools of another service (Slack, Linear) with the token of the right account. It runs inside a Session. A Session is one agent conversation that Router runs from start to end; see «What is a Session» below.

Sources: `acceleration/README.md` and `acceleration/api/openapi.yaml` on the `accelerate` branch, repo `GetStream/athena-ai` at commit `e59acbf`, Slack #video-ai. All checked on September 29.

&#91;embedded content: who calls whom · clients, Router and outside services\]

Router has two parts inside: Model router and sessions. A Session is the agent in one conversation: it calls models through Model router, and calls other services as tools. See «What is a Session» and «How plugins work today» below.

### Router

**What it is.** One Go service (`acceleration/`, runs as `cmd/router`). It has two layers.

1. **Model router** picks a provider and calls its API.
   - The client names a need, not a provider: the string `en-low-latency` («fast, English») or a group name such as `llm-fast`.
   - Router uses its YAML config to turn this string into an ordered list of models. For speech recognition, for example, `deepgram/flux-general-en` comes first. Then it calls the API of the first model in the list.
   - If a provider did not start (no key) or returned an error, Router takes the next model: «routing moves to the next candidate» (README). This is failover.
   - Router writes each call as a row in the `requests` table in Postgres: who called, which provider and model, what it cost. Spend reports are built from these rows (section «What is recorded» in the README).
2. **Agent** is code inside Router that runs a conversation.
   - **Voice mode.**
     1. The agent joins a Stream Video call as one more participant over WebRTC. The `internal/agent/streamedge` package does this with the `getstream-go-webrtc` library.
     2. The agent sends the other people's audio to STT and gets text back.
     3. The text goes to the LLM, and the LLM answer goes to TTS.
     4. The agent gets PCM samples from TTS, encodes them to Opus and sends them to the call (`internal/agent/streamedge/speaker.go:108`).
   - **Text mode.** There is no call: the question comes as text and goes straight to the LLM, and the answer goes back as text. README: «a documentation agent or a support chat is the voice agent with the voice left off».

**What it does, with examples:**

- **One-off model work.** The client sends a recording and gets text, sends text and gets an audio file, asks a question and gets an answer with links, or gets an image from a description.
- **Voice agent.** A support agent joins a Stream call and talks as described above. If a person starts talking while the agent answers, the agent stops speaking.
- **Phone.** Buy a number from Twilio, Telnyx or another vendor, send incoming calls to the agent, call a customer, transfer a call to a human, call a list of numbers.
- **Text agent.** The same agent without voice. If you pass `persist_conversation: true` at creation, Router itself writes the questions and answers as messages to a Stream Chat channel.
- **Agent tools.** This is what the model can call instead of answering with text.
  - Client functions run on the client side.
  - Built-in: knowledge base, memory, sandbox.
  - Tools of other services over MCP. Connectors are about these.
  - The exact call order is in «What is a Session».
- **Money and rules.** Spend report from the `requests` table, app budget, data retention policy.

**What is a Session.** It is one running agent conversation inside Router. It is created with `POST /v1/agents/sessions` and lives until it is closed.

- **Where the idea comes from.** At first the agent was a separate program, `cmd/agent`, and a new process was started for each call. A Session is the same thing, but over HTTP. README: «the process that used to be started per call becomes a session in a process that is already running». The client backend does not start the agent itself; it asks Router to run the conversation.
- **What it holds.**
  - Settings: instructions, models (`llm` and a stronger `subagent` for hard tasks), voice, skills, knowledge base, memory, sandbox, labels for spend tracking. Instead of all this you can point to a saved config with `config_id`.
  - State: the history of turns, the current answer, tool calls in progress.
  - Owner: the client app and, if given, the end user: «The `user_id` claim is also what owns the sessions that end user opens» (`openapi.yaml`). Someone else's Session looks as if it does not exist.

|  | Voice Session | Text Session |
| --- | --- | --- |
| How to create | Pass `call_id` | Pass `"text": true` without `call_id` |
| What happens | The agent joins a Stream call, listens (STT), thinks (LLM) and answers by voice (TTS); you can interrupt it | «No call is joined, nothing is transcribed and nothing is spoken» |
| How you talk to it | By voice in the call, plus `say` and `interrupt` | `respond`; you get `response_delta` and `responded` back |
| Where the history is | The call transcript (`/v1/agents/calls/{id}/transcript`) | Optionally a Stream Chat channel (`persist_conversation`) |
| Examples | Support agent in a call, phone call, voice in Athena | Chat, Playground, Athena chat |

Both kinds share the same logic: «A text session has the same instructions, the same skills… and the same knowledge base as a call would have had» (README).

**How a Session calls tools.** The model can call three kinds of tools.

1. **A client function.**
   1. When creating the Session, the client passes the function name and the JSON schema of its arguments in the `tools` field.
   2. When the model decides to call it, Router sends the message `{"type": "tool_call", "id": …, "name": …, "arguments": …}` to the `…/events` WebSocket.
   3. The client runs the function and answers on the same socket with `{"type": "tool_result", "tool_call_id": …, "output": …}` (`internal/api/sessionws.go:218, 315`).
   4. If no answer comes within `tool_timeout_ms`, the model goes on without the result.
2. **Built-in tools:** knowledge base search, memory, sandbox, handing a task to a stronger model. Router runs them itself.
3. **A tool of another service through a connector (AI-816).**
   - **Connection** is a row in the Router database for one account of a service, for example the company Slack bot or John's personal Slack. It holds an encrypted token.
   - **Binding** is an entry in the agent config: which tools of the service the agent may use and which connection to call them with.
   - In `fixed` mode the binding points to one connection right away.
   - In `session` mode the connection is set when the Session is created, in the `connector_bindings` field.

**Example: one call in a voice Session.**

1. The client backend calls `POST /v1/agents/sessions` with `call_id`, and the agent joins the call as a participant.
2. John says: «Message Sarah on Slack». STT returns this text.
3. Router sends the LLM the conversation history and the list of tools. The LLM answers not with text but with a call to the Slack send-message tool, with the arguments «to whom» and «what».
4. Router finds the connection through the binding. With `fixed` it is the company bot chosen in advance. With `session` it is the connection from this Session's `connector_bindings`, that is, John's personal Slack.
5. Router prepares the token and calls the Slack MCP server:
   1. it decrypts the token of this connection;
   2. if the token has expired, it first refreshes it with a refresh request to Slack;
   3. it sets the header `Authorization: Bearer <token>` on the HTTP request to the Slack MCP server (`internal/connectors/runtime.go:194` on the AI-816 branch);
   4. it sends the JSON-RPC call `tools/call` with the tool name and arguments (`internal/mcp/mcp.go:300`).
6. The Slack MCP server sends the message and returns the result.
7. Router gives the result to the LLM, the LLM writes a short answer, TTS turns it into sound, and the agent says it in the call.

**Why this matters for connectors.**

- In `session` mode the personal account is chosen by the Session owner, so the Session must know its user for sure.
- In the browser and in Athena this is known after login.
- In a phone Session there is no verified user: «A verified phone participant is not automatically a verified application user» (`connector-design.md:186`). So by default only app-owned connections are allowed there.

**Endpoints.** The `openapi.yaml` of the `accelerate` branch has 127 of them. Below they are grouped by purpose.

| Group | Main endpoints | Why, example |
| --- | --- | --- |
| Models directly | `POST /v1/search`, `POST /v1/classify`, `POST /v1/stt/recordings`, `POST /v1/tts/recordings`, `POST /v1/image/generations`, `GET /v1/{modality}/stream` | One-off tasks without an agent. `stt/recordings` takes a recording and returns text, `tts/recordings` takes text and returns audio, `search` takes a question and returns an answer with links. `{modality}/stream` is a WebSocket for those who build the agent themselves and take only one model from Router; `{modality}` in the path is the string `stt`, `tts`, `llm` or `sts`. An example for `stt` is in the paragraph below the table |
| Model choice | `GET /v1/{modality}/providers`, `GET /v1/{modality}/routes`, `/v1/router/configs` | The list of providers and whether they answer now. Also the list of models that the string `llm-fast` expands to |
| Sessions | `POST /v1/agents/sessions` (voice: `call_id`; text: `"text": true`), `…/respond`, `…/say`, `…/interrupt`, `PUT …/instructions`, `GET …/events` | The client backend creates a conversation and opens the `…/events` WebSocket. Router sends JSON messages there: `heard` (recognized text and who said it), `response_delta` (the next piece of the answer), `responded`, `tool_call`. The client sends JSON commands on the same socket: `respond`, `say`, `interrupt`, `tool_result` (`internal/api/sessionws.go`) |
| Agent configs | `/v1/agents/configs`, `/skills`, `/knowledge`, `/voices`, `/sync`, `/plugins` | Save an agent under a name: instructions, models, voice. Load a knowledge base from documents or a site. Connect another service: today through plugins, in AI-816 through connectors |
| Phone and outbound calls | `/v1/phone/numbers`, `/v1/phone/calls`, `…/transfer`, `/v1/dispatch`, `/v1/agents/campaigns` | Buy a number, make a call, transfer to a human, call a list |
| Testing and history | `/v1/agents/simulations`, `/v1/agents/calls/{id}/transcript`, `…/timeline`, `/v1/agents/conversations/{cid}/messages`, `/v1/agents/logs` | Run a recorded conversation script before launch. See what was said on a call and what decisions the agent made |
| Money and policies | `/v1/stats/spend`, `/v1/{modality}/stats`, `/v1/policies/app`, `/v1/policies/organization`, `/v1/data/export` | How much was spent and on what, budgets, data policy, export of all app data |
| Browser access | `POST /v1/agents/calls/{id}/token`, `POST /v1/agents/chat-token`, `POST /v1/agents/guests` | Give the browser a token so the user can join a call or a chat with the agent |

**Example: how `/v1/stt/stream` works.** The Python plugin `plugins/stream` uses this socket, class `STT` in `stt.py`. It is for a Vision Agents agent that runs in the client's Python process and takes only speech recognition from Router.

1. `STT.start()` opens a WebSocket to `/v1/stt/stream` and sends the JSON `{"type": "start", "target": <model>, "sample_rate": 16000, …}` as the first message.
2. `STT.process_audio()` gets a chunk of a call participant's audio from the agent (`PcmData`). It converts it to 16 kHz mono and sends the raw PCM bytes as a binary WebSocket message.
3. Router forwards these bytes to the chosen STT provider and sends back the JSON `{"type": "transcript", "text": …, "final": true|false, …}`.
4. The `STT._received()` method reads the `text` and `final` fields and calls `_emit_transcript_event(...)`. From there the client's Python agent works with this text.

The message format for `tts`, `llm` and `sts` is described in `openapi.yaml`, operation `streamModality`.

**How the client proves who it is.**

- The client has a key in two parts: the public `vak_live_…` and the secret `vas_live_…`. Each request carries `X-Api-Key` and a JWT signed with this secret.
- A token is issued either to the client backend (`server: true`) or to one end user (`user_id`). With a user token you see only that user's sessions (`openapi.yaml`, `securitySchemes`).
- Other deployments can use the `proxy` and `noauth` modes (section Authentication in the README).

**What Router depends on:**

- **Model providers:** Deepgram, Cartesia, ElevenLabs, OpenAI, Gemini, Baseten and others. Router calls their HTTP and WebSocket APIs with its own keys from environment variables.
- **Stream Video.** The agent joins a call as a participant over WebRTC, gets the other people's audio and sends its own voice in Opus (`internal/agent/streamedge`).
- **Stream Chat.** Router itself writes messages to a channel of type agent, with two packages. The first is chatlog: the transcript of a Session without persistent conversation, that is, a voice call or a text Session with saving turned off (internal/session/manager.go:437-440). The package says: «stores what was said in a conversation as Stream Chat messages»; why: «A voice call leaves nothing behind once it ends» (`internal/chatlog/chatlog.go`):
  - it creates a message with `SendMessage` (line 595);
  - while the model writes, it updates the message text with `EphemeralMessageUpdate` (line 453);
  - at the end it saves the final text with `UpdateMessagePartial` (line 503).
- **Stream Chat, conversation.** The second package is `internal/conversation`: «persists text conversations and their visible activity in Stream Chat». This is the history of a text Session with `persist_conversation`.
  - For `"text": true` saving is on by default: `spec.PersistConversation = spec.Text` (`internal/api/sessions.go:513`).
  - For voice it is not allowed: «persistent conversations require text mode» (`manager.go:229-231`).
  - The calls are the same as in chatlog: `SendMessage`, `EphemeralMessageUpdate`, `UpdateMessagePartial` (`internal/conversation/conversation.go:1320-1337`).
- **Stream Chat, incoming messages.** You can write to the agent in Stream Chat, and it will answer (`internal/api/messagehooks.go`).
  - Stream sends each new app message to the Router webhook `POST /v1/chat/hooks/stream` (`internal/chat/hooks.go:26`). The request is checked by the Stream signature.
  - Router answers only text in the agent's own channel: channel type `agent` (`chatlog.ChannelType`). It skips messages marked with a source: chatlog wrote them itself (`addressed`, `messagehooks.go:111-125`).
  - If a Session without persistent conversation runs on the channel, that Session answers (found.Ask, messagehooks.go:235). If a Session with persistent conversation runs, the hook does nothing: the backend drives it through respond (messagehooks.go:136-137). If there is no Session, the message goes to a worker (dispatch.AssignMessage, messagehooks.go:171). The answer comes as a message in the same channel.
- **Postgres** stores keys, agent configs, calls and `requests` rows. **Redis** stores provider state and rate limit counters.
- **Phone:** Twilio, Telnyx and six more vendors (`internal/phone`).
- **mem0** for memory between calls, **turbopuffer** for knowledge base search, **Daytona** as the sandbox where the agent runs code.
- **MCP servers of other services**, for example `https://mcp.slack.com/mcp` from `internal/plugins/plugins.yaml`.

**Who calls Router:**

- SDKs in ten languages (`sdks/`) and the plugin `plugins/stream` for Python;
- the dashboard (`dashboard/`, Next.js), the TUI (`tui/`) and the `cmd/*` commands;
- Athena, through its own fork of Router.

### Athena

**What it is.**

- **Purpose.** An internal assistant for Stream employees on the web and iOS: «An internal, conversation-first AI assistant for Stream employees, on the web and iOS» (README of the `GetStream/athena-ai` repo).
- **Where it came from.** On September 15 Thierry asked for «an example internal chat UI for our framework», Nash suggested the name Athena, and the repo appeared the same day (#video-ai). The point is dogfooding: show on ourselves what can be built on our platform.
- **«Our own ChatGPT» compares the interface, not the model.** We have no LLM of our own. The models are third-party (DeepSeek through Baseten, Deepgram, ElevenLabs, Exa, FAL), and all of them are reached through Router. Repo rule: «Do not add a direct-provider chatbot to Athena. Provider credentials stay server-side» (`AGENTS.md:27`).

**What it can do** (`docs/PROGRESS.md` from September 23):

- chat with streaming, model choice and «thinking» level;
- files up to 20 MB;
- canvas, HTML sites, PDF and images;
- pptx, docx and xlsx, made by a background Pi worker in a Daytona sandbox;
- shared conversations, teams, projects and skills;
- connecting third-party apps;
- admin: spend, limits per group, allowed models;
- **voice calls**.

Status: prototype. «Full deployed acceptance remains open» (PROGRESS.md).

**Its own backend or a wrapper over Router? Its own backend.** Athena is a full app, but it hands all work with models, conversations and voice to Router.

| Part | What it does | Where it runs |
| --- | --- | --- |
| `web/` | A React app in the browser on Stream Chat React and Stream Video React, plus a small Node server `web/server.mjs`. After Google login the server puts the Google token into the encrypted HttpOnly cookie `athena_session`. On each browser request it decrypts the cookie and repeats the request to Athena API with the header `Authorization: Bearer <token>` | Vercel, athena-getstreamio.vercel.app |
| `ios/` | A SwiftUI app | — |
| `server/`: Athena API | A Go service. It checks the employee login and stores rights, teams, projects, tasks, approvals and files in its own Postgres. It calls Slack, Linear and Notion itself through its own connectors | Railway, Nash's personal project |
| `worker/pi/` | Background tasks, for example building a pptx. It runs code in a Daytona sandbox and asks the model through the Router WebSocket `/v1/llm/stream` | Railway and a Daytona sandbox |
| Router fork | The same Router with its own changes: calls models, runs sessions, writes messages to Stream Chat | Railway, its own database |

**About the Router fork.**

- Athena does not use the main Router but the branch `codex/athena-pi-attachments` at commit `f6190090`. It has «50 commits not in `accelerate`», and a test merge gave conflicts in 28 files (PROGRESS.md).
- Nash on September 25: «Athena's accelerate fork has issues which needs resolving».

**Which Router endpoints Athena API calls** (`server/internal/.../agent_executor.go`, `runtime_usage.go`):

- `POST /v1/agents/sessions`: a text Session with `persist_conversation` or a voice one with `call_id`;
- `…/respond`, `PUT …/instructions`, `…/interrupt` and the `…/events` WebSocket;
- `WS /v1/llm/stream` for conversation titles;
- `GET /v1/{llm,stt,tts}/stats` and `GET /v1/llm/providers` for spend and the model list.

**How one message flows.**

1. **Login.** The employee logs in with Google. Athena API accepts only @getstream.io accounts: it checks the `hd` field in the Google token (`server/internal/identity/verifier.go:143`). Nash: «everything on Athena is behind GAuth tied to getstream.io».
2. **Send.** The browser sends `POST /v1/conversation-commands` with the message text. The request goes through the Node server from `web/` (`server/internal/api/access.go:144`).
3. **Router call.** Athena API finds or creates a Session for this conversation: `POST /v1/agents/sessions` with `text: true` and `persist_conversation`. Then it opens the `…/events` WebSocket and calls `POST /v1/agents/sessions/{id}/respond` with the body `{"command_id": …, "text": …}` (`server/internal/api/agent_executor.go:524`).
4. **Write to chat.** Router creates a question message and an answer message in the Stream Chat channel. While the model writes, it updates the answer text (calls from `internal/conversation`, see above).
5. **Show.** The browser is subscribed to this channel through the Stream Chat React SDK and shows the new answer text right away.

Martin described this flow in #video-ai on September 28, and it matches the code: «I see we have Athena API, then a Vision Agents API and a Stream API».

### Who calls whom

| Who | Calls whom | Why |
| --- | --- | --- |
| Client apps through the SDK | Router | Models and sessions |
| Dashboard, TUI | Router | Watch calls and set up agents |
| Athena browser and iOS | Athena API | Login, sending messages, files |
| Athena browser and iOS | Stream Chat, Stream Video | Read messages in real time, make calls |
| Athena API | Router fork | Sessions, titles, spend |
| Athena API | Slack, Linear, Notion | Athena's own connectors, bypassing Router |
| Pi worker | Router fork (`/v1/llm/stream`), Athena API | Background tasks in Daytona |
| Router | Model providers, Stream Video and Chat, phone, MCP servers | Everything the agent needs |

### How plugins work today

Today the Router agent calls other services through plugins. A plugin is an OAuth login to a service's MCP server, tied to one saved agent (config). AI-816 replaces plugins with connectors; what exactly changes is in the table at the end of this subsection. Code: branch `accelerate` @ `9473a0a3`, folder `acceleration/internal/plugins/`, checked on September 30.

Do not confuse this with the `plugins/` folder at the repo root, for example `plugins/stream`. It holds Python provider packages for the Python SDK; they have nothing to do with plugins in Router.

**Where plugins run today** (checked October 1, 2026, 02:48 UTC).

|  | What we see | How it was checked |
| --- | --- | --- |
| Environments | Only staging, at `accelerate.gcp.stream-io-api.com`. There is no production: a production release is only planned | the cluster's namespaces and the deploy code in the infra repo |
| Version | `v0.6.9-dev-1480919a`: commit `1480919a` from September 28 on the `accelerate` branch, with plugins. The connectors prototype is not in it | Log of the initContainer `fetch-binary`; `git merge-base --is-ancestor` |
| Connected plugins | None: `agent_plugin_connections` has 0 rows, and no agent config has a non-empty `plugins` | Read-only `SELECT` from a temporary pod in the staging namespace; the pod was removed |
| Usage | In 10 hours of logs: two `GET /v1/agents/configs/{id}/plugins` with an empty answer and no `authorize` | `kubectl logs` of the current pod; there are no older logs, and ClickHouse is not deployed on this cluster |

So plugins are not used anywhere: we can replace them with connectors without moving any data.

**Catalog.** Five services in `internal/plugins/plugins.yaml`: Slack, Calendly, Cal.com, Shopify and Salesforce. The file is built into the binary (`//go:embed`, `catalog.go:11`), so a new service means a new Router build. Shopify and Salesforce have no single address: the URL has `{instance}`, and the client passes the name of its store or org, for example `mystore.myshopify.com`.

**Endpoints** (described in `api/legacy.yaml`, handlers in `internal/api/plugins.go`):

| Request | What it does |
| --- | --- |
| `GET /v1/agents/plugins?q=…` | The catalog; you can search by name and category |
| `GET /v1/agents/configs/{id}/plugins` | Which plugins are connected to this agent and their status: `pending`, `connected`, `failed` |
| `POST /v1/agents/configs/{id}/plugins/{plugin_id}/authorize` | Starts the login and returns `authorize_url`, the provider's consent page |
| `GET /v1/agents/plugins/callback` | The provider sends the browser back here after consent. No auth |
| `DELETE /v1/agents/configs/{id}/plugins/{plugin_id}` | Disconnects the plugin from the agent |

&#91;embedded content: plugins today · connecting and use in a session\]

Step 1 happens once: after it, the tokens sit in a row for the pair «agent + plugin». Step 2 repeats in each Session of this agent, and all its sessions use the same account. Message 6 comes through the browser: the provider redirects it to the Router callback.

**Step 1. Connect, once per «agent + plugin» pair.**

1. The dashboard calls `POST …/plugins/slack/authorize`; for Shopify and Salesforce with `instance_url` in the body.
2. Router finds the authorization server as the MCP spec says (`StartAuthorize`, `oauth.go:69-89`).
   1. It requests `/.well-known/oauth-protected-resource` from the MCP server's domain and takes the first address from `authorization_servers`. If there is no metadata, it takes the domain itself (`oauth.go:240-249`).
   2. It requests `/.well-known/oauth-authorization-server` and gets the addresses `authorization_endpoint`, `token_endpoint` and `registration_endpoint`.
3. Router takes `client_id` from the environment variable `<ID>_MCP_CLIENT_ID`, for example `SLACK_MCP_CLIENT_ID`. If it is not set, Router registers itself through DCR under the name «Vision Agents» and without a secret (`token_endpoint_auth_method: none`, `oauth.go:260-296`). The variable `<ID>_MCP_CLIENT_SECRET` is read but never sent anywhere (`_ = clientSecret`, `oauth.go:106`).
4. Router creates PKCE (S256) and a random `state`. It writes a row to the `agent_plugin_connections` table with status `pending` and the fields `oauth_state`, `code_verifier`, `client_id`, `token_endpoint`, and returns `authorize_url` (`api/plugins.go:83-102`).
5. The browser opens `authorize_url`, and the person agrees on the provider's screen. The provider redirects the browser to `/v1/agents/plugins/callback?code=…&state=…`.
6. `finishPluginLogin` finds the row by `state` and exchanges `code` for tokens. It writes `access_token`, `refresh_token` and `expires_at` as plain text and sets status `connected`. Then it adds the plugin id to the `agent_configs.plugins` column and sends the browser back to `<DASHBOARD_URL>/agents/<config_id>` (`api/plugins.go:130-184`).

**Step 2. Use, in each Session of this agent.**

1. When a Session is created, Router calls `attachPlugins` (`internal/session/manager.go:347`). If the Session is created without `config_id`, no plugins are attached (`plugin_tools.go:18`).
2. Router takes the rows of this config with status `connected`. If the access token expires in less than a minute, Router refreshes it and saves it. A refresh error is only a warning in the log, and the old token is used (`plugin_tools.go:41-56`).
3. `plugins.Open` connects to each MCP server with its own small JSON-RPC client, not the SDK. It sends `initialize` with `protocolVersion: "2025-03-26"`, then `notifications/initialized` and one `tools/list` without paging (`mcp.go:111-139`).
4. The model gets all the server's tools, with no filter, under the names `<plugin>__<tool>`, for example `slack__<name>` (`mcp.go:95-103`).
5. When the model calls such a tool, `pluginRunner` sends `tools/call` with the header `Authorization: Bearer <token>` and joins the text parts of the answer (`plugin_tools.go:77-86`, `mcp.go:151-196`).
6. If a server does not answer, its tools are simply left out of the Session, and the conversation goes on: «a broken Slack login does not refuse the call» (`plugin_tools.go:16`).

**Disconnect.** `DELETE` erases the tokens in the row, sets `deleted_at` and removes the plugin from the config. The token is not revoked at the provider (`internal/store/plugins.go:135-162`).

### Connectors and how they differ from plugins

Connectors are not a specific piece of code but a model: how the agent gets access to another service and under whose account. The main difference from plugins: the service account lives apart from the agent and has an explicit owner. Competitors have the same model: App at Retell, auth connections at ElevenLabs, owner `app` or `user` at Vercel Connect (see «Comparison» and «The middle layer»).

**Four ideas instead of one row.** With plugins everything sits in one row of `agent_plugin_connections`: which service, whose login, and which agent may use it. Connectors split this into four parts.

1. **Connector** describes a service: the MCP server address, the login method, the provider's rules. One per service, for example «slack».
2. **Connection** is one connected account and its secrets. It has an owner: the client app (`app`, for example the company Slack bot) or an end user (`user`, for example John's personal Slack). It is not tied to an agent.
3. **Binding** is an entry in the agent config: which tools of the service the agent may use and which connection to call them with. In `fixed` mode the connection is chosen in advance. In `session` mode it is passed when the Session is created and must belong to the user of that Session.
4. **Credential resolver** is the only place where a tool call gets a valid token. It decrypts the secret, does the refresh, and marks the connection with a «needs a new login» flag. It can sit on top of our own store or a broker (see «Build our own layer or use a broker»).

**Example.** A company connects a Slack bot once as an app-owned connection. The support agent and the sales agent each get their own binding to it, with different tool sets. John connects his personal Slack as a user-owned connection. When Athena creates his Session, the binding in `session` mode takes exactly his connection. With plugins you cannot do this: each agent needs its own login, and you cannot connect a second Slack account to the same agent.

&#91;embedded content: plugins and connectors · the same example two ways\]

Top: the example with plugins. Two agents hold two separate logins to the same Slack, and there is no personal account at all. Bottom: the same example with connectors. The colored boxes are connections, which live apart from the agents.

**Key differences.**

| Question | Plugins today | Connectors |
| --- | --- | --- |
| Whose account | Whoever clicked «connect»; the database does not record it | The owner is recorded explicitly: `app` or `user` |
| One account for several agents | No: each agent has its own login | Yes: one connection, several bindings |
| Several accounts of one service | No | Yes: the company bot and employees' personal accounts |
| When the account is chosen | Once at connect time, forever for the agent | In advance (`fixed`) or when the Session is created (`session`) |
| What the agent can call | All tools of the server | Only the tools listed in the binding |
| Login methods | Only OAuth | OAuth, API key, bearer token, client credentials |
| Who is responsible for the token | The Session at start, with no guarantees | Credential resolver: encryption, refresh, «needs a new login» status, revoke |
| Where tools come from | Only the service's MCP server | An MCP server or an HTTP request with the connection's token (in «Work plan») |

Where the model comes from: the design in `connector-design.md:10` (AI-816 branch) and how competitors solve the same task (the vendor sections).

**The prototype on the AI-816 branch is one option, not an accepted design.** Nash wrote the code in `codex/connector-support` to show the project. On a Zoom call with Kanat he said it can be fully redone; this is not written down anywhere. When this document points to AI-816 code, it is an example of how the model can be built and where it has gaps.

**How the AI-816 prototype builds the model.** Left: code on the `accelerate` branch. Right: branch `codex/connector-support` @ `7855a261`. On the AI-816 branch the `internal/plugins/` folder is gone; the code moved to `internal/mcp/` and `internal/connectors/`.

|  | Plugins today | AI-816 prototype |
| --- | --- | --- |
| Catalog | 5 services: Slack, Calendly, Cal.com, Shopify, Salesforce | 7: the same without Shopify, plus Gong, Linear, GitHub (`connector-design.md:56`) |
| Whose login | One per «config + plugin» pair: unique index `(config_id, plugin_id)` in `migrations/20260901180000_agent_plugins.sql`. Whose account it is, is not recorded | Connection is a separate entity with owner `app` or `user`; a binding ties it to the agent |
| Two accounts of one service | Not possible | Possible |
| Token storage | Plain text in `agent_plugin_connections` | Encrypted with AES-GCM using the key `ROUTER_AUTH_KEK` |
| Refresh | At Session start, with no lock; an error is only logged | Before the request to the MCP server, under `pg_advisory_lock` (`internal/store/connectors.go:323-346`); on error the connection moves to `needs_reauthorization` |
| MCP client | Its own JSON-RPC, protocol 2025-03-26; it does not keep or send the `Mcp-Session-Id` header (`mcp.go:237-263`) | Official Go SDK v1.8.0 (see «MCP in this document») |
| Which tools the model sees | All tools of the server | Only those allowed in the binding; if a tool's schema changed (`schema_digest`), it is hidden |
| Call timeout | None of its own (`mcp.go:151-196`) | 5 seconds by default |
| Outgoing requests | `http.DefaultClient` with no limits (`mcp.go:82-84`) | A client from `internal/egress`, https only (`egress/public.go:48`) |
| `client_secret` | Read from the environment but not sent | From the variable named in the provider's `client_env` in `connectors.yaml` |

**Moving old logins in the prototype.** On the first start of a version with AI-816, Router moves connected Slack, Calendly, Cal.com and Salesforce logins into encrypted app-owned connections. Then it deletes the old table and the `agent_configs.plugins` column (`connector-design.md:44`).

- Each moved connection must be authorized again: the old login did not keep enough data about the provider and the account.
- The binding is created with no tools: they must be allowed for the agent again.
- Shopify is not moved. The move needs `ROUTER_AUTH_KEK`, or Router will not start. You cannot go back (`connector-handover.md`).

**Do we need the move at all (our view).** Most likely not: the repo shows no signs of a production deploy of Router.

- The only hosted Router is `accelerate.gcp.stream-io-api.com`. The code calls it staging: `const staging = "https://accelerate.gcp.stream-io-api.com"` (`examples/routers/stt_realtime_example/main.go:32`) and `STAGING_ACCELERATION_URL` (`sdks/js/tests/live/staging.test.ts:52`).
- **The cluster has only staging** (checked September 30). Router runs on its own cluster, which has only the staging namespace and release. A production release is planned for «later» in the deploy code of the infra repo but not created.
- **Staging runs the `accelerate` branch with plugins.** Version `v0.6.9-dev-1480919a` (log of the initContainer `fetch-binary`) is a commit from `accelerate`; it is not in `codex/connector-support`.
- **Plugins are almost unused on staging.** In 10 hours of the current pod's log (31,184 lines) there are two `GET /v1/agents/configs/{id}/plugins` requests with a 3-byte answer (an empty list) and no `authorize`. There are no older logs: ClickHouse is not deployed on this cluster.
- So the move code is not needed. It is enough to delete the old table. On staging the `agent_plugin_connections` table is empty: 0 rows, and no agent config has plugins (see «Where plugins run today»). Nobody will need to reconnect anything.

### Where connectors fit

- **Router connectors work in any Session**, voice or text. Stream customers with voice agents for business will get them, and so will Athena.
- **Athena has its own connectors today.** They live in `server/internal/connectors/`: Slack, Linear and Notion over OAuth2 with PKCE or with an API key. Actions in connected apps always need approval (`docs/DECISIONS.md`, D18).
- **Why a shared layer.** Nash on September 17: «For both the product agent and the sovereign AI app… both are hacking together connectors». Thierry answered: «Connectors inside your sovereign ai / So we get it right». Nash on October 1: «The last big missing feature on Athena is slack connection and multiplayer but requires needs some connector support first» ([thread](https://getstream.slack.com/archives/C094V4M57NE/p1790865663910469)). So Athena moves to Router connectors (our view): nobody wrote the word «Router», but AI-816 is the only connector work in progress.
- **Voice in Athena is not the same as a phone call (our view).** In Athena the caller is an employee who already logged in with Google, from the browser or iOS. So we know who they are, and they can go through OAuth in the same app. This does not work with a phone caller (see «Personal token in a voice channel»).

## The middle layer: what it is and do we need it

Thierry asked: «Is MCP enough, do we need an inbetween layer like vercel does?». Short answer: we need MCP, but it is not enough. We also need a middle layer, and AI-816 already has one inside Router. The open question is a different one: run this layer ourselves, or use a broker for some providers.

### What MCP does and does not do

- **What an MCP call is.** It is an HTTP `POST` request to the MCP server URL of another service, for example `https://mcp.linear.app/mcp`.
  - The body is JSON-RPC: `{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_issue","arguments":{…}}}`.
  - The header is `Authorization: Bearer <access token>`.
  - The tool list is requested the same way, with the `tools/list` method.
  - In AI-816 `CallTool` sends this request (`internal/mcp/mcp.go:300`), and `AuthorizeRequest` adds the header (`internal/connectors/runtime.go:194`).
- **What the MCP auth spec covers.** How a client gets a token once:
  - it finds the authorization server;
  - it registers with CIMD or DCR;
  - it goes through OAuth with PKCE.
- **What MCP does not cover:**
  - where to keep the refresh token between calls, and how to encrypt it;
  - who refreshes it, and how to stop two processes from refreshing it at the same time;
  - whose account to use in this session: the company's or a specific user's;
  - how to show the user a consent link, and what to do if access was revoked.

This is the job of the middle layer.

&#91;embedded content: the middle layer · Vercel and Accelerate\]

### How Vercel does it: Eve and Connect

At Vercel these are two separate products ([our review](https://github.com/GetStream/Vision-Agents/blob/codex/connector-support/acceleration/docs/eve-connectors-research.md) from September 24, `vercel/eve` @ `15ad358c`).

1. **Eve** is an open-source agent framework. It runs inside the app process.
   - A connection is described by a file in `agent/connections/`. The file has the MCP server URL, a tool filter and the auth method.
   - Eve calls `tools/list` and `tools/call` itself.
   - If there is no token, Eve pauses the agent's turn while the user goes through OAuth.
2. **Vercel Connect** is a Vercel cloud service. It is the «inbetween layer»: «Vercel Connect is an optional managed authorization service supplying provider grants and tokens».
   - It stores the grant, that is, the refresh token the provider issued.
   - It does the refresh itself.
   - It returns a valid access token on request.
   - Price: $3 per 1,000 token requests on Pro ([pricing](https://vercel.com/docs/connect/pricing)).

**What happens in one tool call in Eve** ([Connect and Eve](https://vercel.com/docs/connect/frameworks/eve), [tokens](https://vercel.com/docs/connect/concepts/tokens)).

1. The model decides to call a tool, for example `linear__create_issue`. The name follows the pattern `<connection>__<tool>`.
2. Before the MCP request, the Connect adapter inside Eve asks Connect for a token for the pair «connector + owner». The owner is the app (`app`) or a specific user (`user`).
3. If there is a grant, Connect returns an access token. The adapter caches it in process memory. The cache key is the connector and the request parameters; the adapter asks for a new token before the old one expires.
4. If the user has no grant, Eve pauses the turn and shows a consent link. After consent, the adapter asks for the token again, and the turn goes on.
5. Eve sends `tools/call` to the MCP server with the header `Authorization: Bearer <token>`.
6. The Connect service does the refresh from the stored grant; the app does not need to.

**Two details that matter for us:**

- **The user must already be verified.** A personal token is issued only to someone signed in to the session; otherwise Eve returns `principal_required`.
- **The other option is your own store without Connect.** Eve lets you plug it in with three functions: get a token, start authorization, finish authorization (`getToken`, `defineInteractiveAuthorization`). So even at Vercel, Connect is optional.

How the adapter proves itself to Connect (checked October 1): the Eve adapter «uses your deployment's Vercel OIDC token to request provider tokens», and Connect verifies that token against the connector's project links to confirm the project and environment may request tokens. Outside Vercel, the app passes a Vercel access token through `vercelToken` instead ([Connect and Eve](https://vercel.com/docs/connect/frameworks/eve), [Vercel Connect](https://vercel.com/docs/connect)).

### What matches it in AI-816

| At Vercel | In AI-816 | What it is in code |
| --- | --- | --- |
| Connection file in `agent/connections/` | Connector and agent binding | Provider description in `internal/mcp/connectors.yaml` and a binding entry in the agent config |
| Owner: `app` or a user | `owner` on a Connection | A field of the connection row in Postgres |
| Installation: the chosen provider workspace | The Connection itself: one specific account | A row with `account_id` and an encrypted token |
| Vercel Connect: stores grants and refreshes | Our own store and refresh inside Router | AES-GCM + KEK; refresh under `pg_advisory_lock` (`internal/store/connectors.go:323-346`) |
| `getToken` with your own store | The credential resolver boundary | The place where a broker can be plugged in later (`connector-design.md:409`) |

### Answer to Thierry's question

- **MCP is enough for the tool call itself.** It is not enough for storing and refreshing tokens or for picking the account.
- **We don't need a separate service like Vercel Connect.** AI-816 already has the same layer inside Router.
- **We can add a broker later** (Connect, Nango or another one) behind the resolver boundary, and only for some providers. Agent bindings won't need to change.
- **Whether to add a broker is a data decision**, made after our own layer has run for a month on Slack and Linear (see «Build our own layer or use a broker»).

## Glossary

| Term | Meaning in this document |
| --- | --- |
| Connector | A description of a service: MCP URL, auth modes, provider policy. LiveKit uses this word for channels, which is a different thing |
| Connection | One authorized account. The owner is the app (app-owned) or the end user (user-owned). It holds encrypted credentials and does not depend on agents |
| Binding (agent binding) | An entry in the agent config: alias, the rule for picking a connection (`fixed` or `session`) and the exact list of allowed tools |
| Grant (tool grant) | Permission for an agent to call one specific tool, pinned by `schema_digest` |
| `schema_digest` | SHA-256 of the MCP tool's name, description and input schema. If the digest changes, the tool is hidden until it is reviewed again |
| Credential resolver | An internal boundary in Router where a tool call gets its token. Input: the verified user, connection, audience and scopes, lifetime. Today our own encrypted store and refresh sit behind it; later a broker can sit there (`connector-design.md:409`) |
| KEK | Key Encryption Key: the key that encrypts tokens in the DB (`ROUTER_AUTH_KEK`) |
| OAuth app | A program registered with a provider: Slack, Google, Salesforce. It is registered in advance and gets a `client_id` and `client_secret` pair. Tokens issued to this app work only with its `client_secret` |
| Consent screen | The provider page «Some app wants access to your account». It shows the OAuth app's name, not the name of the product the user is in |
| Access token | A short-lived key for calling the provider API; at Linear it lives 24 hours. In AI-816 `AuthorizeRequest` puts it in the `Authorization` header of the MCP request (`internal/connectors/runtime.go:194`) |
| Refresh token | A long-lived key used to get a new access token. The provider issues it to one OAuth app, and only someone with that app's `client_secret` can exchange it ([RFC 6749 §6](https://www.rfc-editor.org/rfc/rfc6749#section-6)) |
| Client credentials | An OAuth grant where a service acts on its own behalf, with no user |
| BYO (bring your own OAuth client) | The platform's customer registers an OAuth app with the provider and gives us its `client_id` and `client_secret`. The consent screen then shows the customer's brand |
| Managed app | An OAuth app registered by the platform (us or a broker) and shared by all its customers |
| White-label | The consent screen and the URL show the customer's brand, not the platform's |
| Lock-in | Moving to another platform is technically costly. Example: tokens were issued to our OAuth app, so after a move every user of the customer goes through the consent screen again |
| MCP | Model Context Protocol: an open protocol for tool calls; details in [What MCP is](#mgrnb5r1kqa.175730). The current revision is [2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog) |
| MCP authorization | The OAuth profile from the [MCP spec](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization): discovery (RFC 9728, RFC 8414), authorization code with PKCE, `resource` (RFC 8707), client registration with CIMD or DCR |
| PKCE | Proof Key for Code Exchange (RFC 7636): protects the authorization code from being stolen |
| DCR | Dynamic Client Registration (RFC 7591): the client registers with the provider automatically. Deprecated in MCP 2026-07-28 |
| CIMD | Client ID Metadata Document: the `client_id` is a URL with a JSON description of the client. It replaces DCR in MCP |
| Confused deputy | An agent with one person's rights runs another person's request and shows them data that is not theirs |
| CCaaS | Contact center as a service: Genesys, Five9, Amazon Connect, NICE |
| BYOC | Bring Your Own Carrier: a Genesys mode where an outside carrier provides the phone lines |
| Channel vs tool | Channel: a person writes or talks to the agent through a service (Slack, WhatsApp, iMessage), and the agent answers there; this needs incoming events and a link from the chat to a session. Tool: the agent calls the service API itself. AI-816 connectors are tools |
| MSP | Messaging Service Provider: a provider approved by Apple, through which a business connects to Apple Messages for Business |
| IVR | Interactive Voice Response: the voice menu and queue of a phone line («press 1…»). IVR deflection moves a caller from the queue to a chat |
| RCS | Rich Communication Services: a carrier message format, the next step after SMS. Unofficial iMessage providers fall back to it or to SMS if the recipient has no iMessage |
| Plugin (in Router) | The current mechanism on the accelerate branch: OAuth sign-in to a service's MCP server, tied to one agent config (internal/plugins/). AI-816 replaces it with connectors. Not the same as the Python packages in the plugins/ folder at the repo root |

## Scope and method

We compare five voice agent platforms in detail: LiveKit, Pipecat, Retell AI, ElevenLabs Agents and PolyAI. There is one question: how does an agent on the platform get access to another service, and who handles auth. Seven more voice and CX platforms (Vapi, Bland, Synthflow, Google CES, OpenAI Realtime, Amazon Connect, Cognigy), enterprise agent platforms, assistants and IDEs, auth brokers and iPaaS are covered briefly in «Other products». The «Comparison» tables cover only the five platforms.

- **Sources.** Primary only: vendor docs, API schemas and SDKs, source code (LiveKit and Pipecat are open source). A search snippet does not count as a source.
- **Date.** Pages about the five platforms were opened on September 28, 2026; pages about other products, incidents and the fact check on September 29, 2026; pages about iMessage on September 30, 2026. Vendors change their docs often, so the facts are tied to these dates.
- **Three levels of confidence.** «Documented»: the vendor says so. «Seen in code/schema»: read in the source or SDK. «Our view»: our interpretation, always marked.
- **Closed platforms.** For Retell, ElevenLabs and PolyAI we can't see the internals. We describe the contract, not how it works inside.
- **What we did not do.** We did not create accounts, go through OAuth or measure latency. Performance claims come only with a link to the vendor.

We compare against the connectors model, its prototype on the `codex/connector-support` branch (AI-816) and its design doc `acceleration/docs/connector-design.md`.

### Evaluation criteria

We look at each vendor along eight axes. The first four describe the design; the last four describe behavior under load and at the edges.

| Axis | Question | Why it matters for us |
| --- | --- | --- |
| Integration model | Native connector, webhook function, MCP or developer code? | Sets how much provider-specific code we will maintain |
| Channel or tool | Does the agent talk to the user inside the service (Slack, WhatsApp) or call its API? | These are different products: a channel needs events and session routing, a tool needs a token and a call |
| Authentication | Which modes (API key, bearer, OAuth2), who runs consent, who refreshes tokens? | The hardest part of the task: Nash said «MCP layer needs a review (for auth as well)», Thierry asked «Is MCP enough?» (links in «Why this document») |
| Credential owner | Workspace/app, end user, or the developer in their own code? Can there be several accounts for one provider? | Decides whether we can build personal agents (Athena) and multi-tenant products |
| Tool exposure | How do tools reach the model: allowlist, filters, approval of each call? | The permission boundary: what the agent can do at all |
| Voice runtime | Timeouts, speech while waiting, cancel on interruption, retries | Voice does not forgive long pauses or double side effects |
| Lifecycle | Token expiry mid-call, revocation, reconnect, account change, MCP tool schema change | This is where real integrations break |
| Extensibility | How does a customer add a service that is not in the catalog? | The catalog will never be complete |

The final scalability, maintainability and extendability scores come from these axes. The scale and its reasoning are in the comparison section.

## LiveKit

The LiveKit product called «Connectors» is exactly two voice bridges, **Twilio and WhatsApp**; LiveKit has no Slack in any form. So Nash was right («WhatsApp and Twilio»), not the «Slack and WhatsApp» version. We read the code at [`livekit/agents@57b3227a`](https://github.com/livekit/agents/tree/57b3227a7842697e6ad45b1275369cf9700bf161) and [`livekit/protocol@26ce7b82`](https://github.com/livekit/protocol/tree/26ce7b82f4011a6d77058e3bc1587e7117315a9a).

### What LiveKit Connectors are

- **What is in it.** The proto `service Connector` has five RPCs: `DialWhatsAppCall`, `ConnectWhatsAppCall`, `AcceptWhatsAppCall`, `DisconnectWhatsAppCall`, `ConnectTwilioCall`. The `ConnectorType` enum is `{WhatsApp=1, Twilio=2}` ([proto](https://github.com/livekit/protocol/blob/26ce7b82f4011a6d77058e3bc1587e7117315a9a/protobufs/livekit_connector.proto#L25-L41), [overview](https://docs.livekit.io/telephony/connectors/)).
- **Launch.** Announced on September 1, 2026, with the promise «more on the way… including contact center suites» ([blog](https://livekit.com/blog/introducing-livekit-connectors)). Same price as SIP: $0.004/min on Ship, $0.003/min on Scale.
- **Limits.** Voice only, LiveKit Cloud only (no self-hosted), closed service: only the protos are public.
- **WhatsApp.** It handles only the media. The developer handles the Meta webhooks with SDP and the signature checks. If `ConnectWhatsAppCall` comes too late, you get silence and a dropped call ([WhatsApp](https://docs.livekit.io/telephony/connectors/whatsapp.md)). The proto has no text RPCs.
- **Twilio.** `ConnectTwilioCall` returns a WebSocket `connect_url` for TwiML `<Stream>`. The request has no field for Twilio credentials ([proto](https://github.com/livekit/protocol/blob/26ce7b82f4011a6d77058e3bc1587e7117315a9a/protobufs/livekit_connector_twilio.proto#L25-L65)). The docs themselves suggest Elastic SIP Trunking instead ([Twilio](https://docs.livekit.io/telephony/connectors/twilio.md)).
- **Credentials are not stored.** `whatsapp_api_key` is sent in every RPC, marked `SENSITIVITY_SECRET`: «held in memory for the duration of the call and are never stored» ([blog](https://livekit.com/blog/introducing-livekit-connectors)).

### Channels and tools

- **Channels.** WebRTC clients, SIP trunks (quickstarts for 6 carriers) and Connectors. Every channel joins the room as a participant, so the agent code is the same ([telephony](https://docs.livekit.io/telephony/llms.txt)).
- **Tools.** `@function_tool`, MCP, built-in tools of LLM providers, RPC to the frontend, and HTTP/MCP tools in the no-code Agent Builder ([tools](https://docs.livekit.io/agents/build/tools/), [builder](https://docs.livekit.io/agents/start/builder.md)).
- **No SaaS integrations.** The `livekit/agents` tree has 0 paths with `slack` or `whatsapp`; in [llms-full.txt](https://docs.livekit.io/llms-full.txt) Slack shows up only as the community Slack. The provider-specific code is 77 plugins for models and avatars.

### MCP (`livekit-agents/livekit/agents/llm/mcp.py`)

- **Transports.** `MCPServerStdio` and `MCPServerHTTP` (SSE or streamable HTTP). If `transport_type` is not set, streamable is chosen only for the `/mcp` path ([L343-416](https://github.com/livekit/agents/blob/57b3227a7842697e6ad45b1275369cf9700bf161/livekit-agents/livekit/agents/llm/mcp.py#L343-L416)). Python only ([MCP](https://docs.livekit.io/agents/logic/tools/mcp/)).
- **Auth is `headers` only.** The internal `_create_http_client` takes `auth: httpx.Auth`, but no public parameter passes it. OAuth from the MCP SDK is not available ([L388-405](https://github.com/livekit/agents/blob/57b3227a7842697e6ad45b1275369cf9700bf161/livekit-agents/livekit/agents/llm/mcp.py#L388-L405)).
- **Changing the token.** A public `headers` setter ([L382-386](https://github.com/livekit/agents/blob/57b3227a7842697e6ad45b1275369cf9700bf161/livekit-agents/livekit/agents/llm/mcp.py#L382-L386)) lets you swap the token during a session, but the docs don't mention it.
- **Filtering.** `allowed_tools`, `MCPToolset.filter_tools(fn)`. There is a beta `ToolSearchToolset` (BM25) for large catalogs ([toolsets](https://docs.livekit.io/agents/logic/tools/toolsets.md)).
- **Timeout.** `client_session_timeout_seconds=5` applies to every request; the example raises it to 120 s.
- **No reconnect and no `list_changed` handling.** The tool list is cached until `setup(reload=True)` ([L151-189](https://github.com/livekit/agents/blob/57b3227a7842697e6ad45b1275369cf9700bf161/livekit-agents/livekit/agents/llm/mcp.py#L151-L189)). A server that fails to connect at start is only logged, and the agent starts without it.

### Credentials

- **LiveKit Cloud Secrets.** Encrypted env vars or files per deployment. Changing a value triggers a rolling restart ([secrets](https://docs.livekit.io/deploy/agents/secrets/)). In Agent Builder they are inserted as `{{secrets.ACCESS_TOKEN}}`.
- **User tokens** are passed through job metadata or participant attributes. Getting and refreshing them is the caller's job ([external data](https://docs.livekit.io/agents/logic/external-data.md)).
- **Missing:** OAuth, refresh, a per-user vault and an account registry.

### Call runtime

- **An interruption does not kill the tool:** «waiting for function call to finish before fully cancelling» ([generation.py L1070-1080](https://github.com/livekit/agents/blob/57b3227a7842697e6ad45b1275369cf9700bf161/livekit-agents/livekit/agents/voice/generation.py#L1070-L1080)).
- **Cancel only by explicit opt-in** with `ToolFlag.CANCELLABLE`: «most tools (orders, writes, payments) aren't safe to interrupt» ([async tools](https://docs.livekit.io/agents/logic/tools/async/)).
- **Long calls.** Progress with `ctx.update()`, fillers with `ctx.with_filler()`.
- **Duplicates.** `on_duplicate` takes `allow|reject|replace|confirm`, default allow; about comparing arguments (duplicate\_scope=name\_and\_args) the code warns: «fails open… don't rely on this as an exactly-once guarantee» ([tool\_context.py L178-201](https://github.com/livekit/agents/blob/57b3227a7842697e6ad45b1275369cf9700bf161/livekit-agents/livekit/agents/llm/tool_context.py#L178-L201)).
- **Errors and retries.** A `ToolError` goes to the LLM; any other error becomes «An internal error occurred». No general retries, `max_tool_steps=3`.

### Assessment (our view)

|  | Pros | Cons |
| --- | --- | --- |
| Architecture | Clear split: channel = room participant, tool = function or MCP | A new channel needs changes to the protocol and the Cloud service; only LiveKit can make it |
| Auth | Connectors store no credentials, so the attack surface is small | No OAuth (not even MCP OAuth), no refresh, no per-user tokens |
| Runtime | The best voice tool runtime among competitors: cancellable, progress, fillers, duplicate protection | MCP is Python only, with no reconnect and no schema refresh |
| Scalability | Provider logic lives in MCP servers | Credentials in env or metadata; one MCP connection per job (our view, from the examples) |
| Maintainability | Very little provider code: the MCP client is about 600 lines | All the complexity goes to the developer |
| Extendability | A tool is a function, MCP, or an HTTP tool in Builder | Channels can't be extended from outside |

## Pipecat

Pipecat has no connectors layer and no auth layer: it is a framework with a strong runtime for tool calls, and the developer passes in all credentials. We read the code at commit [`pipecat@94e4090`](https://github.com/pipecat-ai/pipecat/tree/94e40901655e9e082b44c15059c0b3e65b856f37).

### Integrations: channels and tools

- **Catalog.** 196 integrations in 19 categories: the Pipecat team maintains 131, the community 65 ([Supported Services](https://docs.pipecat.ai/api-reference/server/services/supported-services.md)). All of them are models, media and telephony (LLM 29, STT 33, TTS 54), not SaaS.
- **Channels** are real-time media only: 12 transports (Daily, LiveKit, SmallWebRTC, WebSocket, WhatsApp and others) and phone serializers (Twilio, Telnyx, Plivo, Exotel, Genesys, Vonage). There are no text channels (Slack, SMS, WhatsApp messages).
- **Tools** are developer functions (`FunctionSchema`, direct functions) and an MCP client ([function calling](https://docs.pipecat.ai/guides/learn/function-calling)). Both repos have 0 matches for `slack|hubspot|salesforce|calendar|oauth` (git tree search).
- **Slack is not supported**, neither as a channel nor as a tool ([llms.txt](https://docs.pipecat.ai/llms.txt)).

### MCP

- `MCPClient` in `src/pipecat/services/mcp_service.py` supports stdio, SSE and Streamable HTTP ([docs](https://docs.pipecat.ai/server/utilities/mcp/mcp)).
- **Auth is static `headers` only** (or `env` for stdio). The code has no OAuth, no refresh and no `list_changed` handling; the httpx client is created without an `auth=` hook (L106–113). Our view: MCP OAuth is not supported.
- **Filters.** `tools_filter` is an allowlist of names. `tools_output_filters` post-processes output. `tools_arguments` sets fixed arguments that are hidden from the model and override its values (L528–560). This matches `provided_arguments` from our Eve research.
- **Limits.** Only `properties` and `required` are copied from `input_schema`. Only `TextContent` is taken from the result; the rest is dropped (L525–597). The tool list is requested once.
- **Lost connection.** The model gets the error text, the session is reset, and the next call reconnects. The failed call is not retried (L573–628).

### Credentials

- All keys are constructor arguments or env vars. The owner is always the developer.
- **Pipecat Cloud secret sets** are key-value sets per user or organization, mounted as env vars. A changed value needs a redeploy, and a set lives in one region ([Secrets](https://docs.pipecat.ai/pipecat-cloud/fundamentals/secrets.md)).
- **No managed OAuth, token storage or refresh.** End-user tokens can only be passed in the free-form `body` of the `/start` endpoint ([start](https://docs.pipecat.ai/api-reference/pipecat-cloud/rest-reference/endpoint/start.md)); after that it is developer code (our view).

### WhatsApp and Twilio

- **WhatsApp is voice calls only.** Meta webhooks and WebRTC per call, Graph API `/{phone_number_id}/calls` with a bearer token. A temporary Meta token lives less than 2 hours ([WhatsApp](https://docs.pipecat.ai/pipecat/features/whatsapp.md)).
- **Signature checks are off by default.** HMAC `X-Hub-Signature-256` is optional, and the reference example does not turn it on: whatsapp\_secret is not passed (`pipecat-examples/whatsapp/server.py` L102–104).
- **Twilio** connects through Media Streams over WebSocket, or through Daily with SIP. Checking `X-Twilio-Signature` is left to the developer ([production](https://docs.pipecat.ai/pipecat/deployment/telephony-in-production.md)).

### Call runtime (`llm_service.py`)

- `function_call_timeout_secs` is `None` by default, so there is no timeout. It can be set per tool with `@tool_options(timeout_secs=…)`. On timeout the handler is cancelled and the LLM is told the call failed (L313–357, L1794–1812).
- `cancel_on_interruption=True` by default: an interruption cancels the call. With `False` the call becomes async, and the result is added later (L186–189).
- **Protection against speculation.** Calls requested during eager inference are withdrawn, because a side effect can't be undone (L1594–1603). A good model for us.
- There are no tool call retries, so the framework does not repeat side effects. Idempotency is the developer's job.

### Assessment (our view)

|  | Pros | Cons |
| --- | --- | --- |
| Architecture | One neutral tool schema is translated into 14 LLM adapters | No concept of connection, owner or grant |
| Auth | — | Static secrets only, changed by redeploy, nothing per user |
| Runtime | Timeouts, async, cancel on interruption, protection against speculation | The MCP tool list is fixed at start |
| Scalability | Each bot is its own process with its own MCP connection | One secret set per organization and region, no tenant model |
| Maintainability | Rare integrations go into community packages (65 of 196) | SaaS connectors are left entirely to the developer |
| Extendability | A new tool is one async function with a docstring | Anything that needs consent or identity is built from scratch |

## Retell AI

Retell has the cleanest resource model of all competitors: a connection is a separate App object, and each tool is bound to one App. But the auth itself is weak: most often you paste a token, and MCP works without OAuth. Sources: the docs and [retell-python-sdk](https://github.com/RetellAI/retell-python-sdk) @ `75a6a5a` (v6.0.1).

### Catalog: marketing vs reality

- The [retellai.com/integrations](https://www.retellai.com/integrations) page lists about 70 integrations. Most are marked «unofficial»: they are recipes through Zapier or a custom function ([Keap](https://www.retellai.com/integrations/keap)).
- **About 11 providers are really managed** ([overview](https://docs.retellai.com/integrations/overview.md)):
  - CRM: HubSpot, Salesforce, Dynamics 365, GoHighLevel, Zoho;
  - support: Zendesk;
  - calendars: Calendly, Cal.com;
  - knowledge bases: Google Drive, OneDrive, Notion (sync only, no tools).
- **Channels:** phone over SIP, web call, chat widget, SMS (Twilio numbers only, [SMS](https://docs.retellai.com/deploy/enable-sms.md)). WhatsApp and Slack are not documented as channels, and the SDK has no `whatsapp` string.
- **Zapier, Make, n8n** work the other way: they start Retell calls with an API key and are not agent tools ([Zapier](https://www.retellai.com/integrations/zapier)).

### The App model (connection)

- **API.** `POST /create-app`, `PATCH /update-app/{id}`, `GET /list-apps`, `DELETE /delete-app/{id}` (with `force_delete`), `GET /list-app-usages/{id}`, `POST /test-app-auth/{id}` (`src/retell/resources/app.py:98-385`).
- **Fields.** `provider`, `type`, `auth_config` (union `oauth2 | api_key | basic`, «stored encrypted at rest»), `tenant_id`, `tenant_url`. Limit: 20 Apps per provider (`resources/app.py:71`).
- **Status.** `connection_status` is `not_connected | connected | error` and is set by the server. It becomes `error` when the provider rejects the credentials (`app_response.py:133-140`).
- **Usage tracking.** `list-app-usages` shows which agents and versions use an App. You can't delete an App that is still referenced unless you pass `force_delete`.
- **Several accounts.** Each tool is bound to one connection, so two HubSpot accounts are kept apart explicitly ([overview](https://docs.retellai.com/integrations/overview.md)).
- **The owner is always the workspace**: «every agent in your workspace can use its tools». Per-user OAuth is not documented.

### How providers authenticate

| Provider | Credential | Refresh |
| --- | --- | --- |
| [Cal.com](https://docs.retellai.com/integrations/cal-com.md) | API key, «never-expiring» | Not needed |
| [Calendly](https://docs.retellai.com/integrations/calendly.md) | Personal access token | Manual, Reconnect |
| [HubSpot](https://docs.retellai.com/integrations/hubspot.md) | Private app token. Yet the overview says «OAuth»: the docs contradict themselves | Manual, 7-day grace period |
| [Salesforce](https://docs.retellai.com/integrations/salesforce.md) | OAuth client credentials through the customer's External Client App and a «Run As» user | No refresh tokens: «The client credentials flow doesn't issue refresh tokens». A 401 or `INVALID_SESSION_ID` sets `error`, then Reconnect (checked 29.09) |
| [Zendesk](https://docs.retellai.com/integrations/zendesk.md) | Interactive OAuth, `read write` | Retell stores the token. If the provider rejects it: `error` and Reconnect. The refresh mechanism is not described (checked 29.09) |
| [Google Drive](https://docs.retellai.com/integrations/google-drive.md) | Interactive OAuth, `drive.file` | Retell stores the refresh token |

For most providers (GoHighLevel is the exception), a missing scope goes unnoticed: «missing scope or permission doesn't flag the connection» ([overview](https://docs.retellai.com/integrations/overview.md)).

### Integration tools, custom functions, MCP

- **Integration tools.** Each one has a connection, inputs (fixed or filled by the LLM), response variables for chaining, speech while waiting, and a test button on live data ([integration tools](https://docs.retellai.com/build/single-multi-prompt/integration-tools.md)).
  - The timeout is fixed: «3 to 14 seconds, depending on the provider». You can't change it.
  - The public agent schema in SDK 6.0.1 has no such tool type. Our view: it is set up only in the dashboard.
- **Custom function (`type: "custom"`).** Fields: `url`, `method`, `headers`, `parameters` (JSON Schema), `timeout_ms` (1–600 s, default 120 s), `max_retry` (0–5, default 0) with an explicit idempotency warning (`llm_create_params.py:813-930`).
  - Requests are signed with `X-Retell-Signature` (HMAC-SHA256, 5-minute window, [secure webhook](https://docs.retellai.com/features/secure-webhook.md)).
  - Private IPs and metadata addresses are blocked ([custom function](https://docs.retellai.com/build/single-multi-prompt/custom-function.md)).
- **MCP client.** Streamable HTTP transport. «doesn't run an interactive OAuth authorization flow… obtain the access token outside of Retell» ([MCP](https://docs.retellai.com/build/single-multi-prompt/mcp.md)).
  - Config: `Mcp{name, url, headers, query_params, timeout_ms}`. Here `timeout_ms` is the connect timeout, not the call timeout (`llm_create_params.py:1291-1305`).
  - Tools are added explicitly, and `input_schema` is captured when you bind them. How the system reacts to a schema change is not documented (our view).
- **Per-call auth.** Dynamic variables like `{{access_token}}` are put into headers and URLs. You can pass them when you create the call, from an inbound webhook, or with `POST /v2/update-live-call/{call_id}` ([dynamic variables](https://docs.retellai.com/build/dynamic-variables.md)).
  - Retell itself warns that these variables are stored «in plaintext» ([code tool](https://docs.retellai.com/build/single-multi-prompt/code-tool.md)).

### Assessment (our view)

|  | Pros | Cons |
| --- | --- | --- |
| Architecture | App is a separate resource with status, usage tracking and delete protection; a tool is bound to one connection | Integration tools are not in the public schema, so you can't manage them as code |
| Auth | Encrypted storage; secrets are not returned by the API | Tokens are mostly pasted by hand, refresh is manual, no per-user OAuth and no MCP OAuth, a missing scope goes unnoticed |
| Runtime | Retries with an idempotency warning, speech while waiting, SSRF protection | Fixed 3–14 s timeouts on integration tools |
| Scalability | The generic workarounds (HTTP, MCP, Code) don't depend on catalog size | Workspace tokens don't let you build personal agents |
| Maintainability | A small managed catalog means little per-provider code | Retell engineers write every new provider; there is no external manifest |
| Extendability | Custom function, Code tool (QuickJS) and MCP cover everything else | Auth in the workarounds is fully the customer's job |

## ElevenLabs Agents

ElevenLabs has the most mature credentials layer of all competitors. Auth connections have a type, a status and a list of dependents; MCP tools have hash-based approval. But all of this is per workspace: there is no OAuth for the end user. Sources: the docs and the public [OpenAPI](https://api.elevenlabs.io/openapi.json).

### Tools, integrations, channels

- **Tool types.** Webhook, client, system, MCP and code; code is enterprise only ([tools](https://elevenlabs.io/docs/eleven-agents/customization/tools)). The OpenAPI has no code tool schema. Native integrations use a separate type, `api_integration_webhook`.
- **20 native integrations** ([llms.txt](https://elevenlabs.io/docs/llms.txt)):
  - calendars: Cal.com, Calendly, Google Calendar;
  - CRM and support: Salesforce, HubSpot, Zendesk, Freshdesk, Intercom, ServiceNow, Jira;
  - contact center: Genesys;
  - search: Exa, Tavily, Parallel;
  - other: Google Drive, Cursor, Slack, Telegram, Twilio SMS, Custom Channel.
- **Channels: the widest set of all competitors:**
  - widget and SDK;
  - telephony (Twilio, SIP, Vonage, Telnyx, Plivo);
  - CCaaS (Amazon Connect, Genesys, Five9);
  - WhatsApp: both messages and calls ([WhatsApp](https://elevenlabs.io/docs/eleven-agents/whatsapp));
  - SMS;
  - Slack: chat in channels and threads ([Slack](https://elevenlabs.io/docs/eleven-agents/customization/integrations/slack));
  - Custom Channel: an inbound webhook and a signed reply ([custom channel](https://elevenlabs.io/docs/eleven-agents/customization/integrations/custom_channel)).
- **An integration can be both a channel and a tool.** WhatsApp gives a Send Message tool to agents in other channels. Zendesk and Salesforce events can start a conversation (triggers).

### Webhook tools: one envelope for all tools

- **Voice policy.** Webhook, MCP and integration tools share the same fields:
  - `response_timeout_secs`: 5–300 s, default 20;
  - `interruption_mode`;
  - `pre_tool_speech` (`auto` takes recent latency into account);
  - `execution_mode` (`immediate | post_tool_speech | async`);
  - `tool_error_handling_mode`;
  - `tool_call_sound`.
- **Each parameter has exactly one value source:** the LLM (`description`), `dynamic_variable`, `constant_value`, `is_system_provided` or `is_omitted`. The server checks `allowed_values` (`LiteralJsonSchemaProperty`). This is our `provided_arguments`, which we already have.
- **Responses.** `assignments` with `sanitize` save values from the response and hide them from the LLM. `response_filter` trims the response.
- **Environment variables.** Strings, secrets or an auth connection, with different values per environment ([env vars](https://elevenlabs.io/docs/eleven-agents/integrate/environment-variables)).

### Auth connections

- **A separate resource,** `/v1/workspace/auth-connections`. Types: `oauth2_client_credentials`, `oauth2_jwt`, private-key JWT, `basic_auth`, `bearer_auth`, `custom_header_auth`, `mtls`. Some types appear only in API responses (you can't create them through the API): `refresh_token_auth`, `api_integration_oauth2_auth_code`, `slack_bot_auth`, `whatsapp`.
- **Status** is `active | refresh_failed | revoked | credential_invalid`. The «OAuth token-manager refresh path» writes `refresh_failed` and `revoked`. `credential_invalid` is set when a response matches `failure_signatures`.
- **Dependents.** Each connection has `used_by`. Secrets and connections are put in «on egress and exclusively in the headers» ([code tools](https://elevenlabs.io/docs/eleven-agents/customization/tools/code-tools)).
- **Each native integration connects in its own way:**
  - Google Calendar: OAuth ([gcal](https://elevenlabs.io/docs/eleven-agents/customization/integrations/google_calendar));
  - HubSpot: a `pat-` token, US only ([HubSpot](https://elevenlabs.io/docs/eleven-agents/customization/integrations/hubspot));
  - Salesforce: client credentials ([Salesforce](https://elevenlabs.io/docs/eleven-agents/customization/integrations/salesforce));
  - Zendesk: the ElevenLabs OAuth app, your own OAuth client, or an API token ([Zendesk](https://elevenlabs.io/docs/eleven-agents/customization/integrations/zendesk)).
- **Per-user credentials only through `secret__*` dynamic variables:** «never sent to an LLM provider» ([dynamic variables](https://elevenlabs.io/docs/eleven-agents/customization/personalization/dynamic-variables)). They come at session start or from the conversation initiation webhook ([personalization](https://elevenlabs.io/docs/eleven-agents/customization/personalization)). End-user OAuth consent is not documented.

**How the 20 native integrations connect** (checked on September 29 against [llms.txt](https://elevenlabs.io/docs/llms.txt)):

- **OAuth through the ElevenLabs app**: only four: Google Calendar, Google Drive, Slack and Zendesk.
- **API key or personal access token:** Cal.com, Calendly, Cursor, Exa, Parallel, Tavily, HubSpot (`pat-`), Intercom.
- **Basic:** Freshdesk, Jira, ServiceNow.
- **OAuth client credentials through the customer's app:** Genesys, Salesforce.
- **The platform's own tokens:** Telegram, Twilio.
- **Custom Channel secrets.**

### MCP

- **Transports:** SSE (default) and streamable HTTP, HTTPS only.
- **Auth:** `secret_token`, `request_headers`, `request_meta` and `auth_connection`. MCP-spec OAuth (authorization code, DCR) for your own MCP servers is not documented ([MCP](https://elevenlabs.io/docs/eleven-agents/customization/tools/mcp)).
- **Approval policy:** `auto_approve_all | require_approval_all | require_approval_per_tool`, default `require_approval_all`.
- **Schema drift protection.** Each tool stores a `tool_hash`: a SHA256 of its parameters and description. When it changes, the status becomes `needs_review`, and the approved version stays in `approved_definition`. This is a direct match for our `schema_digest`.
- **Approval at runtime.** The client gets `mcp_tool_call` with `awaiting_approval` and a 300 s timeout ([asyncapi](https://github.com/elevenlabs/packages/blob/main/packages/types/schemas/agent.asyncapi.yaml)). How to approve on a phone call or in WhatsApp, where there is no client, is not documented.
- **Availability.** MCP is off by default and is not available in Zero Retention and HIPAA modes.

### Runtime and observability

- **Parallel calls.** The strictest `interruption_mode` among the calls wins ([interruptions](https://elevenlabs.io/docs/eleven-agents/customization/tools/tool-configuration/tool-interruptions)).
- **Call history.** `GET /v1/convai/tools/{id}/executions` returns `latency_secs` and `error_type`. Error types (a string; the values are listed in the field description, it is not an enum): `customer_config`, `customer_auth`, `external_server`, `client_timeout` and others.
- **Not documented:** webhook tool retries, rate limits, idempotency, what happens when a token is revoked mid-call.
- **Docs and schema disagree.** An example uses `approval_policy: "always_ask"`, which is not in the enum; there is no code tool schema; system tools differ between the docs and the schema.

### Assessment (our view)

|  | Pros | Cons |
| --- | --- | --- |
| Architecture | One envelope with a voice policy for all tool types | No public API for the integration catalog and connections |
| Auth | Typed auth connections with a status and `used_by`; secrets are added on egress | No per-user OAuth and no MCP OAuth; the host refreshes `secret__*` tokens |
| Runtime | Timeouts, async, pre-speech, call history with error types | What happens when a token is revoked mid-call is not documented |
| Scalability | A connection is reused across agents and environments | Not built for B2B2C with an account per end user |
| Maintainability | One model for statuses and the envelope | 20 providers are maintained by hand: scopes, regions, triggers, deprecations |
| Extendability | Webhook with an auth connection and MCP cover rare services | Only ElevenLabs can add a full integration |

## PolyAI

PolyAI builds integrations from three parts: a declarative HTTP connector (the APIs tab), a secret store with per-agent access, and Python functions. A PolyAI representative often builds the auth-heavy integrations by hand. The strongest part of the stack is contact centers. Sources: [docs.poly.ai](https://docs.poly.ai/llms.txt) and [ADK](https://polyai.github.io/adk/); marketing claims are marked separately.

### Integrations and channels

- **Categories in the docs** ([integrations](https://docs.poly.ai/integrations/introduction)):
  - telephony: Five9, NICE CXone, Twilio, Amazon Connect, Genesys, Zendesk Talk, Dialpad;
  - CRM: Salesforce;
  - hospitality: OpenTable, Tripleseat, HotSOS;
  - healthcare: Epic;
  - knowledge;
  - MCP.
- **About 25 integrations have pages.** Marketing claims «130+ ready-made connectors» ([poly.ai/integrations](https://poly.ai/integrations)), but the docs don't back that number. The HubSpot page returns 404.
- **Channels:** voice (SIP/CCaaS), SMS/RCS, web chat and chat handoff.
  - SMS goes through PolyAI's own Twilio setup: «No customer Twilio account is needed» ([SMS](https://docs.poly.ai/messaging-channel/sms-and-rcs/sms.md)).
  - WhatsApp is only a label in the UI, with no docs.

### Four ways to build an integration

1. **Python functions (Tools).** They run in a sandbox with `requests`, `urllib3` and `jsonschema` installed, «no additional installs» ([libraries](https://docs.poly.ai/tools/import-library.md)).
2. **APIs tab: a declarative HTTP connector** ([APIs](https://docs.poly.ai/integrations/api/introduction)).
   - You describe a base URL per environment (Sandbox, Staging, Live) and a list of operations as «method + path».
   - A function calls it as `conv.api.salesforce.get_contact("123")`.
   - In the ADK this is the `config/api_integrations.yaml` file, with an `auth_type` per environment: `none | basic | apiKey | oauth2`. Credentials «never appear in the YAML or in function/flow code» ([ADK](https://polyai.github.io/adk/reference/resources/api_integrations/)).
3. **MCP client** ([MCP](https://docs.poly.ai/mcp/agent-studio-integrations.md)).
   - Discovery by URL; each tool is turned on and off on its own.
   - HTTPS only, timeout 1–30 s (default 10).
   - Auth: a header, a query parameter or OAuth client credentials, all through the Secrets Vault.
4. **Managed services**: integrations PolyAI builds itself «with your PolyAI account manager»: Custom SIP, Stripe, PCI Pal, Google Sheets and others ([managed](https://docs.poly.ai/integrations/managed-services.md)).

**Connect Portal** appears only in marketing: «Managed OAuth 2.0», health checks, field mapping. The docs don't mention it.

### Credentials

- **Secrets Vault** ([secrets](https://docs.poly.ai/secrets/introduction.md), [access control](https://docs.poly.ai/secrets/how-to-access-control.md)).
  - Secrets are stored per account. Each secret has its own access list: the agents that can use it.
  - Code reads them with `conv.utils.get_secret(...)`. Changes «take effect immediately».
- **OAuth in the APIs tab is client credentials only**, with automatic refresh. Authorization code and per-user OAuth are not documented.
- **Each integration authenticates in its own way:**
  - Salesforce: the customer gives a Client ID/Secret plus the integration user's login and password, and PolyAI creates the token itself ([Salesforce](https://docs.poly.ai/integrations/salesforce.md));
  - Epic: the customer adds the PolyAI app as trusted, SMART on FHIR ([Epic](https://docs.poly.ai/integrations/epic.md));
  - Tripleseat: self-serve OAuth only on the PLG template ([Tripleseat](https://docs.poly.ai/integrations/tripleseat.md));
  - Amazon Connect: the customer's IAM role with `sts:AssumeRole` ([Amazon Connect](https://docs.poly.ai/integrations/voice/amazon-connect/amazon-connect.md)).
- **User identity in chat.** Verified Context Injection: an HS256 JWT from the customer's backend, tied to `context_id`, read-only in code ([VCI](https://docs.poly.ai/messaging-channel/web-chat/verified-context-injection.md)). But API calls always run as a service account.

### Contact centers and handoff

- **Call transfer.** SIP REFER, INVITE or BYE; custom `X-` headers with variables; `conv.call_handoff(destination, reason, utterance)` ([handoffs](https://docs.poly.ai/voice-channel/handoffs.md)).
- **The Handoff API** is read-only; a record is found by conversation ID or by the CCaaS session ID (`shared_id`) ([Handoff API](https://docs.poly.ai/api-reference/handoff/introduction.md)).
- **CCaaS setup.** Genesys through BYOC on a multi-tenant trunk; Five9 over SIP with `X-PolyAi-Auth-Token` ([Genesys](https://docs.poly.ai/integrations/voice/sip/genesys.md), [Five9](https://docs.poly.ai/integrations/voice/sip/five9.md)).

### Runtime

- **Timeout.** Without delay responses, a function is cut off after 10 s ([delay control](https://docs.poly.ai/tools/delay-control.md)).
- **Async results.** The External Events API takes results from outside into a live conversation ([events](https://docs.poly.ai/api-reference/external-events/introduction.md)).
- **Retries are documented only for outbound webhooks:** up to about 5.5 hours, deduplicated by `X-PolyAI-Event-ID`. `conv.api` has no retries.
- **PCI.** Payments go to PCI Pal: «PolyAI never stores or has access to card details» ([PCI Pal](https://docs.poly.ai/integrations/pci-pal.md)).

### Assessment (our view)

|  | Pros | Cons |
| --- | --- | --- |
| Architecture | A declarative API connector with auth per environment; the same code runs in sandbox and live | Each integration has its own auth pattern |
| Auth | Account-level secrets with per-agent access | Client credentials only; Salesforce needs a service user's password |
| Runtime | Delay control, External Events, mature handoff | 10 s timeout, no retries and no circuit breaking for `conv.api` |
| Scalability | Multi-tenant trunks for CCaaS | Many integrations need a PolyAI employee, so scale is limited by headcount |
| Maintainability | YAML in the ADK, so connectors are versioned | Built-in integrations are hand-written function code |
| Extendability | Customers can use the APIs tab and MCP on their own | A new library or managed service only through a PolyAI representative |

## Comparison

None of the five voice platforms offers managed OAuth on behalf of the end user, or OAuth for MCP. This is a choice that fits their market (see «Why competitors built it this way» at the end of this section); auth brokers are covered in «Other products» and «Build our own layer or use a broker». All facts in the table come from the sections above; the Accelerate row comes from `acceleration/docs/connector-design.md` and `connector-handover.md`.

&#91;embedded content: four paths from a tool call to the provider API\]

The four paths differ in who holds the token and on whose behalf the call runs. Each voice platform supports several paths: Retell, for example, uses both the second and the third. None of the five does the fourth path; AI-816, the brokers, and the connectors in Claude and ChatGPT chose it.

### Facts

| Platform | Integration model | Auth modes | Credentials owner | End-user OAuth | MCP auth | Schema drift protection | Text channels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LiveKit | Code or MCP; 2 voice connectors | Static headers, env secrets | Developer | No (token in job metadata) | Headers only | No, the list is cached | No |
| Pipecat | Code or MCP | Static headers, secret sets | Developer | No (token in `body`) | Headers only | No, the list is fetched once | No |
| Retell AI | \~11 native Apps + custom function, Code, MCP | API key, PAT, basic, OAuth (a few providers) | Workspace (up to 20 Apps per provider) | No (plaintext dynamic variables) | Headers, query, dynamic variables | Not documented | SMS, web chat |
| ElevenLabs | 20 native + webhook, MCP, code | Client credentials, JWT, bearer, basic, mTLS, auth code (native only) | Workspace | No (`secret__*` from the host) | Headers, `auth_connection` | Yes: SHA256 `tool_hash` → `needs_review` | WhatsApp, SMS, Slack, Telegram, Custom Channel |
| PolyAI | Python functions, APIs tab, MCP, managed services | None, basic, API key, OAuth client credentials | Account, with per-agent access | No (JWT only for identity) | Header, query, client credentials | Not documented | SMS/RCS, web chat |
| Accelerate (AI-816) | Connector → Connection → Binding, MCP transport | None, bearer, API key, OAuth2 auth code + PKCE, CIMD/DCR | App or verified user | Yes (API and tests; Volt UI not ready) | OAuth per the MCP spec | Yes: `schema_digest` on each grant | No |

&#91;embedded content: credential path for each platform · 5 competitors and Accelerate\]

The same fact table, laid out by the token's path. The difference is in the second column. In LiveKit and Pipecat the token sits in env or comes from outside with the session. Retell, ElevenLabs and PolyAI store one token per workspace or account. Only the AI-816 design stores and refreshes a user's personal token.

### Assessments (our view)

The assessments take a platform owner's view: what we get if Stream builds it the same way. The «Customer work» column shows how much auth work the platform leaves to its customer.

**Scale (low / medium / high):**

- **Scalability.** High: connections belong to users, several accounts per provider, credentials change without a redeploy. Medium: workspace-level connections are reused, but there are no users. Low: secrets in env or metadata, changed by a redeploy, code or a vendor employee.
- **Maintainability (for the platform owner).** High: almost no provider-specific code. Medium: a small catalog with a shared auth and status model. Low: each integration has its own auth, or staff must do manual work.
- **Extendability.** High: the customer adds a service (HTTP or MCP) on their own and gets auth from the platform's store, with refresh. Medium: they can add one, but must bring the token. Low: only the vendor can add one.

| Platform | Scalability | Maintainability | Customer work | Extendability | Basis |
| --- | --- | --- | --- | --- | --- |
| LiveKit | Low | High | All | Medium | No auth layer by design; secrets change through a rolling restart |
| Pipecat | Low | High | All | Medium | Same; secrets change with a redeploy |
| Retell AI | Medium | Medium | Medium | Medium | Workspace-level App; the token is mostly pasted by hand; custom function and MCP take the token from plaintext variables, not from the App |
| ElevenLabs | Medium | Medium | Medium | High | Auth connections with statuses and refresh; webhook and MCP take them from the store; the customer brings user tokens |
| PolyAI | Low | Low | Medium | High | The APIs tab and MCP take auth from the Secrets Vault, including client credentials with auto-refresh; a PolyAI representative connects some integrations |
| Accelerate (AI-816) | — | — | — | — | Not rated: it is a design, not a product in production. Goal: high scalability and low customer work. Cost: maintaining our own catalog; 2 of 7 providers were tested live |

**Our view.** There is no best option until the scenario is clear:

- **Frameworks** are the cheapest to maintain because they leave auth to the customer. That is right for an open-source library.
- **Among platforms, ElevenLabs has the most balanced model.** It beats Retell on extendability: generic tools take the token from the store. It beats PolyAI on scalability and maintainability: a shared status model instead of manual work.
- **For the first stage in the TL;DR** (customers' voice agents), this is a sensible model to follow.
- **For Athena**, the models are outside this table: brokers, and the Claude and ChatGPT connectors.

### Why competitors built it this way

Each «gap» in the competitors has a likely reason. Vendors don't publish their motives, so the «Reason» column is our assessment. The «Based on» column holds facts from their docs and from our own experience with AI-816.

| Decision | Who | Reason (assessment) | Based on |
| --- | --- | --- | --- |
| Don't store third-party tokens at all | LiveKit, Pipecat | A framework doesn't own the user's identity: the developer already has their own backend, DB and auth. Storing other people's tokens widens the area of responsibility, and people who self-host the framework don't need it | LiveKit advises passing access tokens through job metadata ([external data](https://docs.livekit.io/agents/logic/external-data.md)); credentials for Connectors are «never stored or logged» ([blog](https://livekit.com/blog/introducing-livekit-connectors)) |
| A workspace service account instead of personal accounts | Retell, ElevenLabs, PolyAI | The caller is not a user of the customer's Salesforce or Zendesk; the agent acts for the business. The caller can't go through OAuth during a call | Retell Salesforce runs as a «Run As» user; PolyAI runs as an integration user and checks the caller through ID&V. Our design says the same: «OAuth must complete before an ordinary voice call» (`connector-design.md:425`), «A verified phone participant is not automatically a verified application user» (`:186`) |
| Paste a token instead of an OAuth flow | Retell (Cal.com, HubSpot, Calendly), ElevenLabs (HubSpot, Cal.com) | No need for an own OAuth app at the provider, a review, or refresh. A service account token lives long. Cost: manual rotation | Cal.com API key is «never-expiring»; HubSpot has a 7-day grace period on rotation (Retell section) |
| Call the customer's backend as the generic path | Retell, ElevenLabs, PolyAI | Credentials stay with the customer, who already has auth for their own users. A leak does less damage, and the platform is not responsible for refresh. Cost: work for the customer | The `X-Retell-Signature` signature; Retell's warning not to put secrets in variables but to «use a custom function on your backend» ([code tool](https://docs.retellai.com/build/single-multi-prompt/code-tool.md)) |
| No MCP authorization | All five | The spec is still changing, and servers migrate slowly. Waiting for it to settle is reasonable | In July 2026 DCR was deprecated in favor of CIMD, and revision 2026-07-28 removed sessions from the transport (`connector-design.md:135-139`, [changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)) |
| A small catalog, maintained by hand | Retell (\~11), ElevenLabs (20), PolyAI (\~25) | Each provider has its own OAuth quirks, scopes, regions and deprecations. A wide catalog is an ongoing team cost, not one-time work | Our experience confirms it: the Slack bug with `scope` and `user_scope`, Gong's unknown client auth method, no stable account id in Slack and Linear; 2 of 7 providers were tested live (`connector-handover.md`) |

**Where their choice really is weaker (assessment).** When a customer does need access on behalf of their user, they must get and refresh the tokens themselves and pass them through variables. Retell stores these variables in plaintext. In Retell a missing scope goes unnoticed. These weaknesses don't mean their customers often need this: we have no customers yet, and no data on demand either.

## Other products

This section covers products outside the five voice platforms: other voice and CX platforms, enterprise agent platforms, assistants and IDEs, auth brokers and iPaaS. Incidents and provider quirks are in «Edge cases».

All pages were opened on September 29, 2026. The rules for sources are the same as in «Scope and method». Salesforce help pages return only JavaScript. We then opened them with a headless browser and through the Salesforce repo on GitHub, see «Appendix: fact check». Sierra, Decagon and Parloa keep their developer docs behind a login. What is public is in the same section.

**Main point (our view).** Who owns the credential depends on the channel, not on the vendor.

- Where the user sits in a browser or an IDE, personal OAuth is the standard.
- In voice and contact centers it is a service account, even at Microsoft, Google and Salesforce.
- Platforms that support both scenarios pick the owner per tool or per connection. This is the same as `owner` and binding in AI-816.

&#91;embedded content: who owns the credential · 35 products by class\]

Notes on the diagram:

- **Cognigy** gives personal sign-in only in webchat.
- **Claude.ai**, besides personal OAuth, supports static organization headers (beta). This is «the organization's credential, not a person's» ([authentication](https://claude.com/docs/connectors/building/authentication)).
- **Nango** ties a connection to a user or an organization with `tags` ([tags](https://nango.dev/docs/guides/auth/connection-tags-configuration-metadata.md)).
- **Composio** supports SHARED connections ([shared](https://docs.composio.dev/docs/extending-sessions/shared-connections)).

### Other voice and CX platforms

| Platform | Owner | OAuth in the product | MCP and its auth | Timeout and retries | Notable |
| --- | --- | --- | --- | --- | --- |
| [Vapi](https://docs.vapi.ai/server-url/server-authentication) | Organization; per-call override with `assistantOverrides.credentials` | Google and Slack in the dashboard; client credentials for your own tools | `shttp`, only URL and headers | 20 s (1–300); no retries by default, when on — on any non-2xx | The credential is not sent if the server URL came from the request |
| [Google CES / Dialogflow CX](https://docs.cloud.google.com/dialogflow/cx/docs/concept/playbook/tool) | Service account; user token from variables (`EndUserAuthConfig`) | For tools: «Only client credential grant is supported» | Streamable HTTP; the bearer is not sent on `tools/list` | Sync 30 s, async 60 s | One OpenAPI tool is one operation |
| [OpenAI Realtime](https://developers.openai.com/api/docs/guides/realtime-mcp) | The caller's app | No | `authorization` «not persisted or returned»; the tool list loads async | `timeoutMs` in the Agents SDK | `connector_id` is deprecated for models after September 1, 2026 |
| [Amazon Connect](https://docs.aws.amazon.com/connect/latest/adminguide/ai-agent-mcp-tools.html) | Admin, security profile | Through AgentCore: client credentials, 3LO, OBO | AgentCore Gateway, MCP 2025-03-26 | 30 s | One gateway is one MCP server |
| [Cognigy](https://docs.cognigy.com/ai/for-developers/extensions) | Project Connection; personal sign-in only in webchat | MCP OAuth2 with client credentials fields | SSE URL, tool cache for 10 minutes | Extensions: 20 s | A deleted Connection hangs the flow |
| [Bland](https://docs.bland.ai/agents/tools.md) | Secrets for the whole organization | For native integrations | No MCP client | 1–60 s, up to 4 retries | A secret reference in the body is sent as a literal, but the builder test passes |
| [Synthflow](https://docs.synthflow.ai/mcp-actions.md) | Workspace | HubSpot, Salesforce | No auth or bearer | — | An unreachable MCP server is skipped silently |

Vapi schemas were checked against the OpenAPI at `https://api.vapi.ai/api-json`. The Decagon security page says: «Short-lived JWT tokens… discarded after each session» ([security](https://decagon.ai/security)). There are no details.

- **The host brings the personal token.**
  - Vapi accepts an access and refresh token in the call request. Google CES takes the token from session variables. OpenAI does not store the token.
  - Our view: this is a fourth path to a personal token. The customer stores and refreshes the token; the platform only puts it into the call. Accelerate should keep this path too, for customers who already have their own vault.
- **A generic HTTP tool uses the token of a native connection.**
  - In Synthflow, the Authentication Type of your own HTTP action can be Salesforce, Salesforce Sandbox, HubSpot or GHL. ElevenLabs auth connections work the same way.
  - This supports item P1 «a built-in HTTP tool on top of a connection».
- **Scheduled refresh for non-OAuth secrets.** Bland has a Refresh secret: «a value Bland fetches on a schedule, so the agent never holds an expired token mid-call», every 10 minutes to 24 hours.
- **Retries are off by default everywhere.** When turned on, Vapi retries «for any non-2xx status code». This supports dropping blind retries in AI-816.
- **The tool list may not be loaded yet.** In OpenAI Realtime, «While listing is still in progress, the model can't call a tool that hasn't loaded yet». This is a third state for the row «MCP server unavailable at start», next to «available» and «unavailable».

### Enterprise platforms: AWS, Salesforce, Microsoft, Google

These four platforms model both scenarios explicitly, so they are the closest to our decision.

| Product | Entities | Where the owner is chosen | Consent during the conversation | Voice |
| --- | --- | --- | --- | --- |
| [AWS AgentCore Identity](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-authentication.html) | Workload identity, credential provider, token vault. The workload access token «contains both the identity of the agent and the identity of the end user» | On the function: `@requires_access_token(auth_flow="USER_FEDERATION")` or `"M2M"`. Or on the target in Gateway. There is also `TOKEN_EXCHANGE` | `on_auth_url` + a callback that checks the session; the link lives 10 minutes | Not described: the 142 AgentCore pages say nothing about consent without a browser. 3LO is a browser redirect (checked 29.09) |
| Salesforce Named Credentials | Named Credential → External Credential → Named or Per-User Principal → permission set ([glossary](https://developer.salesforce.com/docs/platform/named-credentials/references/named-credentials-reference/nc-glossary.html)) | On the External Credential, that is, on the integration | For MCP, not at all: «User-level authentication isn't supported» ([considerations](https://help.salesforce.com/s/articleView?id=ai.agent_mcp_considerations.htm&language=en_US&type=5)). For External Credentials, in advance, in personal settings. A Salesforce blog promises consent on the first call, which does not match the docs | Service Agents and Voice run as the agent user ([Agentforce Voice guide](https://github.com/salesforce/einstein-platform/blob/main/resources/afv-implementation-guide/03-build.md)) |
| [Copilot Studio](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configure-enduser-authentication) | Connector, connection, connection reference, DLP | On the tool: end-user or maker credentials, end-user by default | Login card or link, then retry | Teams Phone: «No authentication» |
| [Google ADK](https://adk.dev/tools-custom/authentication/) and Agent Identity | `AuthScheme` + `AuthCredential` → `AuthConfig`; auth manager with auth providers | On the tool or toolset | `adk_request_credential`, popup, automatic retry | Dialogflow CX: no personal OAuth |

- **The AWS model is almost the same as ours, but it has three modes, not two.**
  - Modes: 3LO (personal), 2LO (service) and on-behalf-of token exchange.
  - AWS writes: «Many agent implementations will require all patterns» ([patterns](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/common-use-cases.html)).
  - We do not have the third mode. More in «Personal token in a voice channel».
- **An unverified user id is a known trap.**
  - For `ForUserId`, AWS writes that the platform «does not verify this string».
  - AWS advises blocking this call with IAM and keeping ids separate per identity provider and user ([workload token](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/get-workload-access-token.html)).
  - In AI-816, the `user` owner must also be a verified user.
- **Token revocation cannot be detected.** AWS writes: «Tokens can be revoked… which AgentCore cannot detect». For this there is `forceAuthentication=true`, which also deletes the refresh token ([identity-authentication](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-authentication.html)).
- **Access to the credential itself is also controlled.**
  - Salesforce grants a principal through permission sets and warns: «Per-user resolution inherits access; it doesn't scope it on its own» ([SF blog](https://www.salesforce.com/blog/connect-agentforce-external-mcp-servers/)).
  - In Copilot Studio an admin can block maker credentials for an environment. After that, scheduled agents «fail», and «immediately» ([maker creds](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configure-no-maker-authentication)).
  - AI-816 has no policy for «which agents and users may use this Connection».
- **Who gets the tool list from an MCP server with personal tokens.**
  - In AWS Gateway, the admin goes through 3LO when creating the target, or the schemas are set in advance. The second option is «recommended… when human intervention isn't possible» ([blog](https://aws.amazon.com/blogs/machine-learning/connecting-mcp-servers-to-amazon-bedrock-agentcore-gateway-using-authorization-code-flow/)).
  - Google CES does not send the bearer token on `tools/list`, so the server must return the list without auth ([CES tools](https://docs.cloud.google.com/customer-engagement-ai/conversational-agents/ps/tool/open-api)).
  - Our view: this directly affects which account `schema_digest` is computed for in a `session` binding.
- **ADK stores tokens in session state by default** and itself warns that this «can pose security risks» ([ADK auth](https://adk.dev/tools-custom/authentication/)).

### Assistants and IDEs

| Product | Owner | Admin and user | MCP client registration | Token expired |
| --- | --- | --- | --- | --- |
| [Claude.ai](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities) | Personal OAuth; organization headers (beta) | The owner turns it on; «Each person still needs to authenticate individually» | CIMD, DCR, Anthropic credentials, entered by the customer; no client credentials | Refresh on 401 and «proactively up to five minutes before the stored expiry» ([authentication](https://claude.com/docs/connectors/building/authentication)); if that fails, a Connect card and a retry of the call |
| [Claude Code](https://code.claude.com/docs/en/mcp) | Personal, keychain | Managed servers, allow and deny lists | CIMD, DCR, `--client-id` | Refresh and one retry |
| [ChatGPT Workspace Agents](https://help.openai.com/en/articles/20001143) | End-user or agent-owned | The agent owner picks per app; in Slack, shared only | Same as apps: CIMD, DCR, static | Tries refresh. On `invalid_grant`: «OAuth credentials expired. Reconnect required.», and Desktop has no button for it ([codex#47513](https://github.com/openai/codex/issues/47513)). The official docs do not describe this |
| [VS Code](https://code.visualstudio.com/api/extension-guides/ai/mcp) | Personal. Tokens are in SecretStorage, encrypted with a key from the OS keychain (`dynamicAuthenticationProviderStorageService.ts:184-187` @ `f20366e4`) | GitHub and Entra providers | DCR, CIMD, manual client id | Open bugs with a stale client |
| [Cursor](https://cursor.com/docs/mcp) | Personal, also for team servers: «OAuth is per-user, including for MCP servers shared at the team level» ([cloud agents](https://cursor.com/docs/cloud-agent/capabilities)) | Enterprise allowlist | DCR by default or a static client in `mcp.json`. No CIMD: «still on our radar» (a Cursor employee on the [forum](https://forum.cursor.com/), 22.09.2026) | Tokens are stored with the editor's secrets API (per the built Cursor 3.22.7 code). What happens when a token expires is not described |
| [Gemini CLI](https://geminicli.com/docs/tools/mcp-server/) | Personal, file `~/.gemini/mcp-oauth-tokens.json` | — | DCR | A refresh race wipes the credentials |
| [Glean](https://developers.glean.com/guides/actions/authentication) | OAuth Admin, OAuth User or a service credential | The admin sets it up, the user authorizes | Pre-registered | A service credential does not depend on a session |
| [Dust](https://docs.dust.tt/docs/personal-vs-workspace-credentials-for-tools-mcp-servers) | Shared or personal | The admin picks per tool | The admin does the first OAuth | The agent pauses, the user sees a «Connect account» card, and after connecting the call continues. The event `tool_personal_auth_required` is a «non-terminal event that pauses the workflow» ([source](https://github.com/dust-tt/dust/blob/53637106/front/lib/actions/mcp_internal_actions/events.ts#L67-L73)) |
| [Slack MCP](https://docs.slack.dev/ai/slack-mcp-server/) | Personal token | The admin approves the app and each MCP server | Pre-registered only: «We do not support… Dynamic Client Registration» | Not described in the docs. The `mcp.slack.com` metadata on 29.09 has a `refresh_token` grant. In February 2026 the token lived 1 hour with no refresh ([claude-code#29257](https://github.com/anthropics/claude-code/issues/29257)) |

- **There is a third owner: the agent itself.**
  - ChatGPT Workspace Agents tell apart «End-user account — each person running the agent authenticates with their own account» and «Agent-owned account — the agent uses a shared connection».
  - When deployed in Slack, all of the agent's connections must be shared. If you publish an agent with personal connections, other people can act «as the creator» ([workspace agents](https://help.openai.com/en/articles/20001143)).
  - Glean warns that runs with a user token «fail silently when it expires». A service credential «doesn't depend on anyone's session» ([agent identity](https://docs.glean.com/administration/agent-identity/overview)).
  - Our view: our `owner: app` covers the agent account. But the API and docs should name the cases «agent in a shared channel» and «scheduled agent» explicitly.
- **Changing the owner later is expensive.** In Dust, moving from shared to personal makes everyone authorize again.
- **The MCP 2026-07-28 spec added required client rules** ([authz](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization), [client registration](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/client-registration)):
  - check `iss` in the authorization response (RFC 9207) and store the issuer in the same record as the PKCE verifier;
  - on step-up, request the union of the old and new scopes;
  - bind client credentials to the authorization server's `issuer` (a client id from CIMD works across servers, a client id from DCR does not);
  - «MUST NOT assume refresh tokens will be issued».
  - Checked against the AI-816 code on September 29. `iss` is checked, but it is required only if the server declared `authorization_response_iss_parameter_supported` (`internal/mcp/oauth.go:522-529`). There is no scope union on step-up: `insufficient_scope` is not handled anywhere. Binding the client to `issuer` is partial: the issuer is saved but not checked on refresh (`oauth.go:539-595`). Endpoints are accepted only over https (`internal/egress/public.go:48`).
- **Claude moves away from DCR where it can:** «DCR causes Claude to register a new client on every fresh connection».
  - It uses only the first item of `authorization_servers`.
  - It waits 10 s for discovery, registration and the token endpoint, and 30 s for refresh ([authentication](https://claude.com/docs/connectors/building/authentication)).
- **Lazy auth.**
  - A protected call gets a 401, the user sees a Connect card, and the same call is retried with «no context lost».
  - A `200` response with `isError` does not start sign-in. Step-up starts only on a 403 with `insufficient_scope` ([lazy auth](https://claude.com/docs/connectors/building/lazy-authentication)).
- **Everyone has refresh bugs.**
  - Gemini CLI [#29048](https://github.com/google-gemini/gemini-cli/issues/29048): in a race, an `invalid_grant` response deletes «the fresh valid credentials just saved by the winner».
  - Claude Code [#66210](https://github.com/anthropics/claude-code/issues/66210): an old refresh token was used 4 days after rotation.
  - Claude Code [#89862](https://github.com/anthropics/claude-code/issues/89862): refresh without `scope`, and Entra issues a token with the wrong `scp`.
  - VS Code [#321834](https://github.com/microsoft/vscode/issues/321834): a stale DCR client is not reset on 401.
  - Our view: the advisory lock in AI-816 is justified. Two rules are missing: do not delete the credential on a temporary refresh error, and send `scope` in the refresh request.
- **The redirect URI is fragile.**
  - Cursor replaced `cursor://…` with localhost, and pre-registered apps stopped working ([forum](https://forum.cursor.com/t/oauth-redirect-uri-changed-from-cursor-to-http-localhost-for-streamable-http-mcp/165019)).
  - Anthropic's hosted connectors for M365, Gmail and Calendar work only through the claude.ai redirect.
- **Turning on a connector in the product and getting the provider's consent are different steps.** ChatGPT says: «Provider approval, OAuth scopes, and ChatGPT action settings are separate checks». The Entra admin may be a different person ([admin controls](https://help.openai.com/en/articles/11509118)).

**Anthropic and OpenAI: the product and the API work differently.** The product runs MCP OAuth itself, while the API expects a ready token from the caller. This is an argument for Accelerate being a product with its own token lifecycle.

- **Claude custom connectors** run OAuth themselves (`oauth_dcr`, `oauth_cimd`, `static_headers`, always S256). When they refresh the token is in the Claude.ai row of the table above.
- **Anthropic Messages API, MCP connector:** «API consumers are expected to handle the OAuth flow» ([mcp-connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)). Managed Agents vaults can refresh if you pass a refresh block ([vaults](https://platform.claude.com/docs/en/managed-agents/vaults)).
- **OpenAI Responses `mcp`:** «OAuth client registration and authorization must be handled separately by your application» ([MCP guide](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)). **ChatGPT apps** do CIMD, DCR, PKCE and `resource` themselves ([auth](https://developers.openai.com/plugins/build/auth)).

### Brokers and iPaaS

What the five voice platforms lack, brokers sell as their main product: OAuth on behalf of the end user with automatic refresh. This is the «middle layer like Vercel's» that Thierry asked about. A detailed review of Vercel Eve and Vercel Connect is in [`eve-connectors-research.md`](https://github.com/GetStream/Vision-Agents/blob/codex/connector-support/acceleration/docs/eve-connectors-research.md). The option «adopt a third-party connection broker» is covered in `connector-design.md:515` and postponed («no broker selected»).

| Vendor | Per-user OAuth and refresh | Your own OAuth client (BYO) | Pricing model |
| --- | --- | --- | --- |
| [Vercel Connect](https://vercel.com/docs/connect) | Subjects `app` and `user`; «drives the provider's refresh flow automatically» ([tokens](https://vercel.com/docs/connect/concepts/tokens)) | Vercel Managed or Customer Managed | $3 per 1,000 token requests on Pro; 500 per month on Hobby ([pricing](https://vercel.com/docs/connect/pricing), updated 26.08.2026). Before 25.09.2026 it was $3 per 10,000 ([archive](https://web.archive.org/web/20260626052428/https://vercel.com/docs/connect)), so the price went up 10 times |
| [Composio](https://docs.composio.dev/docs/authentication) | «Authentication is always per user» | Yes, four levels of white-label | $0.0003 per tool call; $0.10 per connection above 1,000 ([pricing](https://composio.dev/pricing)) |
| [Nango](https://nango.dev/docs/guides/auth/token-refreshing) | Refresh at least once every 24 hours; a webhook on failed refresh ([webhooks](https://nango.dev/docs/guides/platform/webhooks-from-nango)) | Recommended; for some APIs there are shared Nango apps (see below) | $0.29 per connection per month + $50 base ([pricing](https://nango.dev/pricing)) |
| [Arcade.dev](https://docs.arcade.dev/en/references/auth-providers) | «The client and the LLM will never see the token» | Yes; shared Arcade apps «share any rate limits» | $0.10 per auth event, $0.01 per tool call ([pricing](https://arcade.dev/pricing)) |
| [Pipedream Connect](https://pipedream.com/docs/connect) | `external_user_id`, managed refresh | «you can always use your own client» | Connect is $99 per month billed yearly, 100 external users included, then $2 each ([pricing](https://pipedream.com/pricing)). Billed monthly it is $150 per month (checked October 1) |
| [Paragon](https://docs.useparagon.com) | «fully managed authentication» | Yes, white-label | Prices are not public, «Talk to sales». Billed per Connected User, which is usually an organization ([pricing](https://www.useparagon.com/pricing), [connected users](https://docs.useparagon.com/billing/connected-users.md)) |
| [Merge Agent Handler](https://docs.merge.dev/merge-agent-handler/how-it-works) | «OAuth refresh happens here» | Yes, with your own brand | 1 credit per tool call ([pricing](https://merge.dev/pricing/agent-handler)) |

The most detailed open catalog is `packages/providers/providers.yaml` in Nango (commit `f140b0e`). It shows what it costs to maintain your own catalog.

- **Size and pace.** 1029 entries in 29,177 lines. Auth modes: `API_KEY` 339, `OAUTH2` 303, `OAUTH2_CC` 106, `BASIC` 100, `TWO_STEP` 66, `MCP_OAUTH2` 32. In 30 days, 65 commits changed the file, mostly with the message «add support for X».
- **A declarative description is not always enough.**
  - 417 of 1029 entries put parameters of a specific connection into the URL.
  - 57 need code: post-connection scripts, credential checks and cleanup before delete.
  - Examples of quirks: Salesforce needs `instance_url` from the token response; Slack needs `authed_user` (user token or bot token) and `disable_pkce: true`; QuickBooks needs `realmId` from the callback; Zoho needs a data center choice.
  - Our view: `connectors.yaml` needs templates from the connection config and fields like `token_response_metadata`, or we will get stuck on Salesforce, Zoho and QuickBooks.
- **A connection test is standard.**
  - Zapier requires it: «Missing required keys: type and test» ([schema](https://github.com/zapier/zapier-platform/blob/03908d0/packages/schema/docs/build/schema.md)).
  - Make has an `info` step; in n8n, 159 of 409 credential files have `test`; in Nango, 299 of 439 API key or basic providers have `proxy.verification`.
- **Refresh: two camps.**
  - Proactive: Nango, 15 minutes before expiry, plus a cron for connections not refreshed for a day.
  - Nango has a 10 s Redis lock, and the code says it is «not a distributed lock». Failures are counted at most once a day, after 4 days the status is `refresh_exhausted`, and a webhook comes both on failure and on recovery ([token refreshing](https://nango.dev/docs/guides/auth/token-refreshing)).
  - Make and Activepieces also refresh 15 minutes before.
  - Reactive (on 401): Zapier (`autoRefresh`) and n8n (30 s lease, 60 s margin).
- **A managed OAuth client or your own.**
  - A managed client shows someone else's brand on the consent screen: «users authorize Nango instead of your product», «Composio wants to access your account».
  - Quotas are shared with the broker's other customers, and the provider can revoke such an app.
  - Most important, you cannot take the tokens with you: «only your own developer app lets you export access and refresh tokens and move off Nango» ([auth guide](https://nango.dev/docs/guides/auth/auth-guide.md)). With Pipedream's own client, user credentials are not given out ([oauth clients](https://pipedream.com/docs/connect/managed-auth/oauth-clients)).
  - Our view: a managed client is a lock-in point.
- **Self-hosted installs do not need the managed client's secret.** Activepieces sends code exchange and refresh through `https://secrets.activepieces.com/refresh`.
- **User verifier.** Arcade checks that «the user who is authorizing the tool is the same user who started the authorization flow». By default Arcade allows your own apps only with your own verifier ([production](https://docs.arcade.dev/en/build/user-facing-agents/secure-auth-production)).
- **Composio SHARED connections** are the closest match to our `owner: app`. They have an ACL (`allowAllUsers`, `allowedUserIds`), and they must be «explicitly pinned» in a session.
- **Nango's license is Elastic License 2.0:** «You may not provide the software to third parties as a hosted or managed service…» ([LICENSE](https://github.com/NangoHQ/nango/blob/f140b0e/LICENSE)). The catalog has no separate license. The npm package `@nangohq/providers` 0.71.10 has no `license` field and no LICENSE file, and neither does `packages/providers/` (checked September 29). We can use the `providers.yaml` format as a model. Whether we can copy the data itself is a question for a lawyer.
- **Your own OAuth client at Google and Microsoft is more than a registration.**
  - Google: restricted scopes need an assessment «at least every 12 months» by an accredited auditor.
  - Microsoft: multitenant apps from unverified publishers after November 8, 2020 are blocked under risk-based consent ([publisher verification](https://learn.microsoft.com/en-us/entra/identity-platform/publisher-verification-overview)).
  - If the client belongs to Stream, Stream carries this load. If it is BYO, the customer does.

## Build our own layer or use a broker

**Decision (assessment).** The main layer is our own, in Router. A broker can sit only behind the credential resolver boundary, and only for some providers. Whether to add one, we decide from data: today 2 of 7 providers are checked live, and the cost of maintaining our own catalog is not measured. Broker prices and features are in [Brokers and iPaaS](#mgrnb5r1kqa.91761).

**A broker does not replace the AI-816 model.** Connection, binding, `schema_digest` and the user owner are needed in any case.

But brokers have ready answers for three things we would otherwise build ourselves:

1. Who owns the OAuth app at the provider: shared or BYO.
2. Alerts on failed refresh.
3. A catalog of hundreds of providers.

**The sensible path is a mix.** Our own code for a small, verified catalog and for MCP servers with MCP authorization. For the long tail of rare providers, an adapter to a broker behind the same boundary. Before choosing a broker, check what `eve-connectors-research.md` lists: tenancy, latency, residency, token export and cost.

**Whose OAuth app sits behind the broker.** This decides who is locked in to whom (diagram in «Whose OAuth app»):

- A broker with its own managed apps locks Stream in to the broker, not the customer in to Stream (third row of the diagram).
- A broker with OAuth apps registered to Stream can be replaced (fourth row).

**Decision rule (recommendation).**

1. We decide whether we need a broker from data, after a month of our own layer running on Slack and Linear: how much work two providers take.
2. In any case, we register OAuth apps to Stream, not to the broker. Then the broker can be replaced without reconnecting users.

Facts about brokers and source links are in «Brokers and iPaaS».

## Whose OAuth app: customer lock-in, brand and risks

This section answers the question: «We are a SaaS. Do we want customers locked in to the platform, and how do others do it?» It is written for people who have not worked with OAuth before; the terms (OAuth app, consent screen, refresh token, BYO, managed app) are in the «Glossary». A short conclusion is at the end; its tasks are in «Work plan» and its questions in «Open questions».

### Where lock-in comes from

Tokens issued to our OAuth app work only with our `client_secret`. Say a customer moves to a competitor. Then each of its users must go through the consent screen again, now for the new app. A hundred employees means a hundred «please reconnect» emails. This is the lock-in: not through a contract, but through the pain of moving (our view, from RFC 6749 §6).

&#91;embedded content: whose OAuth app · 4 options and what happens when a customer leaves\]

How to read the diagram:

- **The first row is the classic lock-in of the customer to us.**
- **In the second row the customer also cannot leave alone:** the tokens are with us, and there is no API to export them.
- **The third row is a trap for us.** If the broker uses its own app, it is Stream that is locked in. To leave the broker, we would have to reconnect every user of every customer.
- **The fourth row is the way out of this trap.** Nango writes that only your own app «lets you export access and refresh tokens and move off Nango» ([auth guide](https://nango.dev/docs/guides/auth/auth-guide.md)).

### What AI-816 already has

All checked on the `codex/connector-support` branch on September 29.

- **Stream's app.**
  - In `internal/mcp/connectors.yaml`, Slack, GitHub and Salesforce have `client_env` (lines 14, 84 and 105).
  - If the customer did not pass its own app, this one is used (`internal/mcp/oauth.go:149-150`).
- **The customer's app (BYO).**
  - The fields `oauth_client_id` and `oauth_client_secret` are «for a manually registered integration» (`api/openapi.yaml:6080-6090`).
  - For Gong only the customer has its own app, mode `customer_dcr` (`connectors.yaml:91`).
- **Tokens can be imported but not exported.**
  - «Credential material is write-only» (`api/openapi.yaml:1498`).
  - The same endpoint `PUT /v1/agents/connections/{id}/credentials` «imports a provider-issued OAuth grant». So you can move to us with your tokens, but you cannot leave us with them.

### What others do

| Who | Default | Way out for large customers |
| --- | --- | --- |
| Nango | Shared apps for some APIs | Recommends your own app: with it you can take the tokens ([auth guide](https://nango.dev/docs/guides/auth/auth-guide.md)) |
| Pipedream Connect | Their app; user credentials are not given out | Your own app, and then you can get the credentials ([oauth clients](https://pipedream.com/docs/connect/managed-auth/oauth-clients)) |
| Composio | Managed: «Composio wants to access your account», shared quotas | Your own app for specific services, four layers of white-label ([managed](https://docs.composio.dev/docs/authentication/custom-app-vs-managed-app)) |
| Arcade | Their apps only with their user verifier | In production you basically need your own ([production](https://docs.arcade.dev/en/build/user-facing-agents/secure-auth-production)) |
| Vercel Connect | Vercel Managed | Customer Managed ([Connect](https://vercel.com/docs/connect)) |
| Activepieces | Their app through a secrets proxy | Your own app or white-label for the platform (`oauth2/index.ts` @ `611db01`) |
| n8n | On n8n Cloud, managed OAuth for some Google services | Self-hosted: your own app only ([google oauth](https://docs.n8n.io/integrations/builtin/credentials/google/oauth-single-service)) |
| ElevenLabs (Zendesk) | ElevenLabs app | Your own app or an API token ([Zendesk](https://elevenlabs.io/docs/eleven-agents/customization/integrations/zendesk)) |
| Claude | Hosted connectors for M365, Gmail and Calendar work only through the claude.ai redirect | Third-party MCP servers: CIMD, DCR or your own client id ([Claude Code MCP](https://code.claude.com/docs/en/mcp)) |

**Common pattern (our view).**

- By default almost everyone gives you their own app. The customer connects in a minute, and soft lock-in appears on its own.
- Almost everyone also gives a way out through the customer's own app.
- Vendors do not publish their reasons. The most likely one: large customers want their own brand on the consent screen, their own security rules, and the ability to leave.

### What it costs us if the app is ours

- **Provider reviews.**
  - Google: Gmail and Drive need a security audit «at least every 12 months» ([restricted scopes](https://developers.google.com/identity/protocols/oauth2/production-readiness/restricted-scope-verification)).
  - Microsoft: without verified publisher status, users may not be able to give consent ([publisher verification](https://learn.microsoft.com/en-us/entra/identity-platform/publisher-verification-overview)).
  - Slack: MCP is allowed only for Marketplace apps and internal apps. Commercial apps outside the Marketplace can call `conversations.history` once a minute (see «For Athena» in «Edge cases»).
  - Our view, from the same Slack rule: a customer's app installed only in its own workspace counts as an internal app, and this limit does not apply to it.
- **Shared limits.**
  - Brokers' managed apps share quotas across all customers. Composio writes «share quota across all Composio users», Arcade writes «share any rate limits… with other Arcade customers».
  - With one Stream app for everyone, a shared ceiling exists only at providers that count the limit per app. Our check found two: Google and Microsoft Graph.
    - **Two providers have a shared ceiling for all customers with one Stream app.** Google counts quotas per Cloud project, for example Calendar allows 10,000 requests per minute per project ([quota](https://developers.google.com/workspace/calendar/api/guides/quota)). Microsoft Graph has a limit of «130,000 requests per 10 seconds» per app across all tenants ([throttling](https://learn.microsoft.com/en-us/graph/throttling-limits)).
    - **At the others, customers do not affect each other.** Slack counts «per API method per workspace/team per app». Salesforce counts per org. HubSpot counts per installing account for a Marketplace app. Linear counts per user.
    - More in «Appendix: fact check».
- **One revocation stops all customers.** Nango warns that a shared app «may be revoked by the provider at any time». If a provider blocks Stream's app, that integration stops working for all customers at once.
- **Liability for a leak.** In the Salesloft Drift case, one vendor's token store gave access to the Salesforce data of hundreds of customers ([GTIG](https://cloud.google.com/blog/topics/threat-intelligence/data-theft-salesforce-instances-via-salesloft-drift)). The more tokens of others we hold, the bigger the target.
- **Brand.** Our customers are developers. Their users will see «Stream wants access» and may not know who that is.

### What it costs us if the app is the customer's (our view)

- **Harder start.** The customer registers the app, sets up the redirect URI and passes the provider reviews on its own.
- **We need guides for each provider.** And support for «why does my app not work» questions.
- **Less soft lock-in.** But only if we allow token export; today we do not.

### Where the lock-in really is (our view)

Lock-in through tokens works, but it is weak and unfriendly: a security review spots it easily. Judging by the table above, brokers themselves leave customers a way out.

Strong lock-in is what is expensive to move, even if the tokens can be taken:

- agent configs and bindings;
- the voice policy for tools;
- the call log and observability;
- tuned prompts and tests.

### Conclusion (recommendation)

By default, use Stream's app: the customer connects fast, and this already exists. Give large customers their own app (BYO) with no artificial limits: the API for this already exists. Register OAuth apps only to Stream, not to a broker.

Keep customers with product value (the list above), not with tokens. Athena does not need provider reviews: it runs on Stream's internal app.

For external customers, Slack is a closed loop. Until an app is in the Marketplace, Slack MCP is not allowed for it, and a review needs at least 10 installs in active workspaces ([changelog](https://docs.slack.dev/changelog/2026/09/01/slack-marketplace-install-requirement/)). So the first external customers connect Slack through their own internal app (BYO) (our view).

Tasks are in [Work plan](#mgrnb5r1kqa.35017); token export and white-label are in [Open questions](#mgrnb5r1kqa.37294).

## Personal token in a voice channel

The main problem with personal connections in voice is consent. OAuth needs a browser, and the caller does not have one. There are three working ways and one dangerous one. The voice channels of large enterprise platforms do not support personal OAuth at all.

&#91;embedded content: three ways to get a personal token when the user speaks by voice\]

**Voice channels of enterprise platforms work without personal auth.** So «voice = service account» is not only a choice of voice startups (our view).

- **Copilot Studio.** The Teams Phone channel requires «No authentication» ([Teams Phone](https://learn.microsoft.com/en-us/microsoft-copilot-studio/voice-teams-phone-agent)). For real-time voice agents: «OAuth and Microsoft authentication aren't supported» ([real-time](https://learn.microsoft.com/en-us/microsoft-copilot-studio/voice-realtime-voice-agents)).
- **Salesforce.** In a channel without login, the agent «runs as the agent user». The caller is checked with a code sent by email, and the result «stores the verified ID in a `VerifiedCustomerId` variable» ([Agentforce Voice guide](https://github.com/salesforce/einstein-platform/blob/main/resources/afv-implementation-guide/03-build.md)). For MCP in Agentforce, «User-level authentication isn't supported» ([considerations](https://help.salesforce.com/s/articleView?id=ai.agent_mcp_considerations.htm&language=en_US&type=5)).
- **Dialogflow CX.** Tools support API key, client credentials, service agent, mTLS, or a bearer from `$session.params`. There is no personal OAuth ([CX tools](https://docs.cloud.google.com/dialogflow/cx/docs/concept/playbook/tool)).

**Linking before the call.**

- AWS AgentCore Consent Portal (blog of September 14, 2026): the user can «grant consent before invoking a tool» and press Disconnect ([blog](https://aws.amazon.com/blogs/machine-learning/manage-end-user-oauth-consent-for-ai-agents-with-amazon-bedrock-agentcore/), [docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-consent-portal.html)).
- With the Salesforce per-user principal, each user signs in ahead of time, in personal settings: «each user authenticates to the external system from the External Credentials page in their personal settings… click Allow Access» ([help](https://help.salesforce.com/s/articleView?language=en_US&id=sf.nc_create_edit_oath_browser_ext_cred.htm&type=5)).
- In Copilot Studio the user can open the connections page to «refresh or revoke» ([end-user auth](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configure-enduser-authentication)).

**A link during the conversation is bound to a browser session.**

- **AWS.** The callback must check that the current user is the same one who started the flow. The user id «should be fetched from the active application session on the user's browser», and the link lives 10 minutes ([session binding](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/oauth2-authorization-url-session-binding.html)).
- **Google ADK.** Finishing the flow needs `userId`, `userIdValidationState` and `consentNonce` «from your session storage» ([ADK Agent Identity](https://adk.dev/integrations/agent-identity/)).
- **Claude.** A protected `tools/call` gets a 401. Claude shows a Connect card and «retries the same tool call automatically» ([lazy auth](https://claude.com/docs/connectors/building/lazy-authentication)).
- **Our view.** An SMS link opened on a phone with no session in our app will fail this check. We need our own signed one-time link, bound to the session and the verified user.
  - None of the platforms we checked document an OAuth link by SMS during a call: Retell, Vapi, Amazon Connect, Twilio, Genesys, NICE, Five9, Talkdesk, Decagon.
  - The closest match is Call companion in Dialogflow CX: «Dialogflow sends an SMS message to the user's phone», and the link opens a Google page for entering data ([call companion](https://docs.cloud.google.com/dialogflow/cx/docs/concept/call-companion)).
  - Another one is Alexa: it sends a «card or push notification prompting the user to link» in the app ([Alexa](https://developer.amazon.com/en-US/docs/alexa/account-linking/how-users-experience-account-linking.html)).

**Token exchange without consent.**

- **AWS AgentCore.** `TOKEN_EXCHANGE` per RFC 8693 and 7523: the user is «not prompted to sign in a second time» ([patterns](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/common-use-cases.html)).
- **Claude Enterprise Managed Auth.** GA since August 24, 2026. The client exchanges an ID token from the corporate IdP for an ID-JAG, and the ID-JAG for an authorization server token (jwt-bearer, RFC 7523). The first IdP is Okta. **Linear** is among the first connectors. DCR does not work with EMA ([blog](https://claude.com/blog/enterprise-managed-auth), [docs](https://claude.com/docs/connectors/building/enterprise-managed-auth), MCP extension [EMA](https://modelcontextprotocol.io/extensions/auth/enterprise-managed-authorization)).

**Device code by voice is an anti-pattern.** RFC 8628 warns about remote phishing (§5.4, [RFC 8628](https://www.rfc-editor.org/rfc/rfc8628)). The Storm-2372 campaign lured victims to «authenticate using a threat actor-generated device code». Microsoft «recommends blocking device code flow wherever possible» ([MSFT](https://www.microsoft.com/en-us/security/blog/2025/02/13/storm-2372-conducts-device-code-phishing-campaign/)).

**Checking the caller.** Caller ID alone does not prove who someone is. Twilio gives the `StirVerstat` attestation only «when the incoming call has SHAKEN PASSporT identity headers» ([Twilio](https://www.twilio.com/docs/voice/trusted-calling-with-shakenstir)). HIPAA §164.312(d) requires to «verify that a person… is the one claimed» ([45 CFR](https://www.law.cornell.edu/cfr/text/45/164.312)). Our design already covers this: «A verified phone participant is not automatically a verified application user» (`connector-design.md:186`).

**Conclusion (recommendation).**

In a phone session, only app-owned connections are available by default. A personal account is connected before the call, on a «connect accounts» page, as AWS, Salesforce and Copilot Studio do. During the conversation, only with a one-time link bound to the session and the user. Tasks are in the [Work plan](#mgrnb5r1kqa.35017).

Token exchange or EMA is not available for Athena now (checked September 29). Stream staff use Google Workspace: #internal-it has notices about 165 Google Workspace licenses and «login with google». And EMA in Claude supports only Okta at launch: «Okta is supported at launch, with support for additional identity providers coming soon» ([blog](https://claude.com/blog/enterprise-managed-auth)); Slack is on its «coming soon» list.

## Edge cases

Most edge cases are not documented by competitors. Only ElevenLabs (credential statuses, schema drift) and the frameworks (cancellation, duplicates) describe the behavior. «—» means «not documented and not visible in the code». The Accelerate column is based on `connector-design.md` and `connector-handover.md`.

| Case | LiveKit | Pipecat | Retell AI | ElevenLabs | PolyAI | Accelerate (AI-816) |
| --- | --- | --- | --- | --- | --- | --- |
| Token expires in the middle of a call | No refresh; a public `headers` setter that docs do not describe | No refresh; headers are set at creation. Test on 29.09: on a 401 the model gets «Error calling mcp tool echo: Server returned an error response». The session is not reset and does not reconnect | `update-live-call` changes variables; whether headers are re-read is not checked | Token manager refreshes, and on failure sets `refresh_failed`; what happens to a live call is — | Auto-refresh only for client credentials | Refresh on every request under an advisory lock; a `needs_reauthorization` checkpoint before refresh |
| Provider revoked access | The error goes to the LLM as `ToolError` | The error text goes to the LLM | `connection_status=error`, manual Reconnect | `revoked` / `credential_invalid` by `failure_signatures` | — | `invalid_grant` → `needs_reauthorization`; revoke on the provider side is not built |
| Several accounts of one provider | Not modeled | Several `MCPClient`, each with its own credentials | Up to 20 Apps; a tool is bound to one | A connection id on each tool (our view, from the schema) | — | Different aliases for different connections, no fallback by order |
| A different account was connected on reconnect | — | — | — | — | — | Rejected if there is a stable account id; Slack and Linear have no id yet |
| An MCP tool schema changed | Cached until `reload=True`, `list_changed` is ignored | The list is fetched once | `input_schema` snapshot at binding (our view) | `tool_hash` → `needs_review` | — | `schema_digest` mismatch → the tool is hidden until review |
| Retry of a call with a side effect | No retries; `on_duplicate` argument check «fails open» | No retries; speculative calls are revoked | `max_retry` is 0 by default, with an idempotency warning | No retries: webhook tools have no such field in the schema, only `response_timeout_secs` | No retries for `conv.api` | No blind retries; code `connector_outcome_unknown` (by design) |
| Default call timeout | MCP 5 s | None (`None`) | Integration tools 3–14 s (fixed); custom 120 s | 20 s (range 5–300) | 10 s | `timeout_ms` on the binding; 5 s by default (internal/session/connector\_tools.go:19); latency not measured |
| User interrupts the agent during a call | Waits for the end; cancels only with `CANCELLABLE` | Cancels (`cancel_on_interruption=True`) | A custom function is not canceled: «runs to completion (up to the timeout)» ([custom function](https://docs.retellai.com/build/single-multi-prompt/custom-function)) | `interruption_mode` per tool | — | The call is canceled (`internal/agent/agent.go:1182-1188`), the MCP server gets `notifications/cancelled`, and the error text goes into the history. If a refresh was running at that moment, the connection stays in `needs_reauthorization` |
| MCP server is down at start | Logged, the agent starts | Reconnects on the next call | The error goes to the agent, the conversation goes on | — | — | Required → the session does not start; optional → a `connector_unavailable` event |
| Missing scope | — | — | Silent: «doesn't flag the connection» | — | — | `connector_scope_required` exists only in `connector-design.md:389`; the code never returns it. `granted_scopes` is stored but not compared with what the tools need |
| Provider rate limits | — | — | — | — | — | Not handled. The `429` in the design (`connector-design.md:387`) is Accelerate's own quota; connectors code does not handle `429` or `Retry-After` from the provider |

Sources for each cell are in the matching vendor section above.

### For Athena

The first three rows block the «agent in a shared Slack channel» scenario, and the AI-816 design does not cover them.

| Case | Accelerate (AI-816) | Competitors or standard |
| --- | --- | --- |
| A shared agent in a group chat answers Bob with Alice's token (confused deputy) | Not covered: the model assumes one subject per session (`connector-design.md:343-348`). We need a rule: in a session with several participants, only app-owned connections, or only data that everyone can see | ElevenLabs Slack: context per thread, access through the workspace bot, no per-user tokens ([Slack](https://elevenlabs.io/docs/eleven-agents/customization/integrations/slack)) |
| Slack allows MCP only for Marketplace apps and internal apps | One operator OAuth client (`internal/mcp/connectors.yaml`, `client_env: SLACK`). A Stream internal app works for Athena. Customers need a Marketplace review or BYO: the API already takes `oauth_client_id` (`internal/mcp/oauth.go:148`) | «Only apps published in the Slack Marketplace and internal apps can use MCP» ([Slack MCP](https://docs.slack.dev/ai/slack-mcp-server/)) |
| Slack rate limits for non-Marketplace apps | Not handled | Since 29.05.2025, new commercial non-Marketplace apps get 1 request per minute and up to 15 objects for `conversations.history` and `conversations.replies`; internal apps are not affected ([changelog](https://docs.slack.dev/changelog/2025/05/29/rate-limit-changes-for-non-marketplace-apps)). MCP has «the same rate limits» as the Web API. But apps that are neither Marketplace nor internal cannot use MCP at all ([Slack MCP](https://docs.slack.dev/ai/slack-mcp-server/)). A Marketplace review needs at least 10 installs in active workspaces ([changelog](https://docs.slack.dev/changelog/2026/09/01/slack-marketplace-install-requirement/)). The related conclusion is in «Whose OAuth app», the check is in «Appendix: fact check» |
| Offboarding: an employee left, but their user connection stays linked | There is only `DELETE` of one connection, and it is a soft delete. `owner_id`, `account_id`, `granted_scopes` and `cached_tools` stay in the row (`internal/store/connectors.go:294-306`). There is no delete by user, no link to deactivation, and no foreign key on `owner_id` (checked 29.09) | Nobody documents it |
| Tokens expire often | Live refresh not tested (`connector-handover.md`, Remaining work item 2) | Linear: access token 24 hours; since April 1, 2026 all OAuth apps have refresh tokens ([Linear OAuth](https://linear.app/developers/oauth-2-0-authentication)). Slack: 12 hours with rotation ([token rotation](https://docs.slack.dev/authentication/using-token-rotation/)). Google: a refresh token in Testing status lives 7 days ([Google OAuth](https://developers.google.com/identity/protocols/oauth2)) |
| Refresh token reuse with rotation | Advisory lock and a checkpoint before refresh (`internal/connectors/runtime.go:25,38`), a test with one pinned revision | RFC 9700 §4.14: reuse revokes the active token ([RFC 9700](https://www.rfc-editor.org/rfc/rfc9700)). Linear has a 30-minute grace period |
| Step-up: a tool needs a wider scope (`insufficient_scope`) | Scope can be widened with a new authorization (`connector-design.md:208`); `insufficient_scope` is not handled during a session | MCP: «Clients acting on behalf of a user SHOULD attempt the step-up authorization flow» ([authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)) |
| Account identity in Slack Enterprise Grid | No stable account id for Slack and Linear yet (`connector-handover.md`) | An org-ready app is installed on the organization and «isn't automatically added to the workspaces» ([org-ready apps](https://docs.slack.dev/enterprise/organization-ready-apps/)) |
| Token audience (RFC 8707 `resource`) | Covered: `resource` is sent (`internal/mcp/oauth.go:259-260`) | MCP: `resource` «MUST be included in both authorization requests and token requests» |
| Prompt injection and swapped tool descriptions | Covered: descriptions and results are treated as untrusted (`connector-design.md:417`), `schema_digest` includes the description | MCP: annotations are untrusted unless the server is trusted ([tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)) |
| SSRF and DNS rebinding for custom MCP URLs | Covered: the address is checked on connect (`internal/egress/public.go`, `connector-design.md:457`) | Retell blocks private IPs; MCP [security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices) |
| Google: restricted scopes | Not handled (Google is not in the catalog, but Nash asked for it) | Restricted scopes need a CASA assessment with yearly re-verification; a limit of 100 refresh tokens per account and client ([restricted scopes](https://developers.google.com/identity/protocols/oauth2/production-readiness/restricted-scope-verification)) |
| KEK rotation or loss | Versioned keyring, lazy rewrap; losing the keys = losing all grants; a drill is in the plan (`connector-handover.md`) | Not documented |
| Incognito and data retention | A commitment for `incognito` in the design (`connector-design.md:497`) | ElevenLabs: MCP is not available in Zero Retention and HIPAA modes |
| If Slack or WhatsApp become channels | Out of current scope (`connector-design.md:24`) | Slack Events: a 2xx reply within 3 seconds, up to 3 retries, header `x-slack-retry-num` ([Events API](https://docs.slack.dev/apis/events-api/)). WhatsApp: a 24-hour service window ([send messages](https://developers.facebook.com/documentation/business-messaging/whatsapp/messages/send-messages)) |

File paths are on the `codex/connector-support` branch, under `acceleration/`.

### Incidents and provider quirks

The cases come from incidents, provider docs and open issues of MCP clients. The «What to do» column is a recommendation, not a fact.

&#91;embedded content: three outcomes of one refresh with rotation\]

The advisory lock in AI-816 already closes the race between two replicas. The lock does not help with a lost response. On reuse, Auth0 «immediately invalidates the refresh token family» ([Auth0](https://auth0.com/docs/secure/tokens/refresh-tokens/refresh-token-rotation)). So the only way to recover is the provider's grace window: [Linear](https://linear.app/developers/oauth-2-0-authentication), [Atlassian](https://developer.atlassian.com/cloud/jira/platform/oauth-2-3lo-apps/), [Okta](https://developer.okta.com/docs/guides/refresh-tokens/main/), [Slack](https://docs.slack.dev/authentication/using-token-rotation/).

| Case | What we know | What Accelerate should do (recommendation) |
| --- | --- | --- |
| Tokens stolen from an integration vendor | Salesloft Drift, August 8–18, 2025: stolen Drift OAuth tokens gave access to customers' Salesforce. GTIG recommends to «Enforce IP restrictions» and not to grant the `full` scope ([GTIG](https://cloud.google.com/blog/topics/threat-intelligence/data-theft-salesforce-instances-via-salesloft-drift)). Gainsight, November 2025: «A token issued in 2017 could theoretically still work in 2025» ([Gainsight](https://www.gainsight.com/blog/how-we-accelerated-a-year-of-security-work-in-weeks/)) | Static egress IPs per region, so a customer can restrict the app by IP. A maximum grant age without new consent. An alert on a spike of calls per connection |
| Provider revoked access, but no 401 yet | Slack sends a `tokens_revoked` event ([event](https://docs.slack.dev/reference/events/tokens_revoked)). Microsoft CAE answers 401 with a claim challenge and can lag «up to 15 minutes» ([CAE](https://learn.microsoft.com/en-us/entra/identity/conditional-access/concept-continuous-access-evaluation)). Google RISC sends a `token-revoked` event, and «Only `refresh_token` is supported». The action: delete the token and ask for consent again ([RISC](https://developers.google.com/identity/protocols/risc), checked 29.09) | Move the connection to `needs_reauthorization` on the event or claim challenge, not only on `invalid_grant` |
| Grant limit per OAuth client | Google: «Limit of 100 refresh tokens per Google Account per OAuth 2.0 client ID» ([oauth2](https://developers.google.com/identity/protocols/oauth2)). Salesforce: «Each connected app allows five unique approvals per user. After a fifth approval is made, the oldest approval is revoked.» Each refresh token counts as a separate approval ([help](https://help.salesforce.com/s/articleView?id=xcloud.remoteaccess_request_manage.htm&type=5), checked 29.09) | When the same account reconnects, update the existing connection instead of creating a new one. Warn if a user has several connections on one OAuth client |
| Refresh response is lost | The provider already issued a new pair, but we still have the old refresh token. Grace windows: Linear 30 minutes, Atlassian 10 minutes, Okta 30 seconds by default, Slack a «short grace period». In AI-816 the connection then stays in `needs_reauthorization`, and credentials are kept. The test `TestRefreshOutcomeSurvivesLostResponsesAndCanceledWorkers` covers this and passes (29.09) | Store the refresh start time in the checkpoint. Inside the grace window, retry the refresh with the old token, and only then ask the user to sign in again |
| Advisory lock boundary | An advisory lock works inside one Postgres database. In AI-816 it is a session `pg_advisory_lock` on the tenant + connection pair, held for the whole refresh (`internal/store/connectors.go:323-346`). Nango locks on Redis, and its code says «not a distributed lock». A customer with their own OAuth app (BYO) may refresh the same grant on their side (our view) | State that a connection lives in exactly one database. Declare unsupported the case where a BYO customer refreshes the same grant somewhere else |
| Temporary error during refresh | Gemini CLI [#29048](https://github.com/google-gemini/gemini-cli/issues/29048): a network error wipes credentials. In AI-816 a network error gives `ErrOAuthRefreshUncertain`: the connection stays in `needs_reauthorization`, credentials are not deleted (`internal/connectors/runtime.go:97-112`). Refresh runs on the tool call's context, so an interruption or the 5 s timeout during refresh gives the same result | Do not delete the credential on a network error (AI-816 already does this). Run refresh on a separate context, not the call's context. On a network error, retry first instead of asking for reauthorization right away |
| Refresh without `scope` | Claude Code [#89862](https://github.com/anthropics/claude-code/issues/89862): Entra issues a token with a different `scp`, and the server answers 401. AI-816 does not send `scope` on refresh (`internal/mcp/oauth.go:543-548`) | Send `scope` in the refresh request |
| Malicious `authorization_endpoint` in metadata | CVE-2025-6514 in mcp-remote, CVSS 9.6: command injection when opening the URL ([JFrog](https://jfrog.com/blog/2025-6514-critical-mcp-remote-rce-vulnerability/)). MCP best practices say clients «MUST reject `javascript:`, `data:`, `file:`…» ([security](https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices)) | Already done in AI-816: all endpoints go through `egress.ValidatePublicHTTPSURL` (`internal/egress/public.go:48`). Side effect: endpoints with a query string are also rejected |
| Reads and writes across trust boundaries in one session | GitHub MCP, May 26, 2025: prompt injection in a public issue leaked private repos ([Invariant](https://invariantlabs.ai/blog/mcp-github-vulnerability)). Supabase MCP, July 2025: an agent with `service_role` leaked tokens into a ticket ([General Analysis](https://generalanalysis.com/blog/supabase-mcp-blog)). Asana MCP: data was visible to other orgs from May 1 to June 4, 2025 ([BleepingComputer](https://www.bleepingcomputer.com/news/security/asana-warns-mcp-ai-feature-exposed-customer-data-to-other-orgs/)) | A `read_only` option on the binding. Ask for confirmation if a session has both a tool with untrusted content and a write tool |
| Provider terms on data use | Slack API ToS: you may not «use API Data to train a large language model» ([ToS](https://slack.com/terms-of-service/api)). Google Workspace allows only the user's «personalized model» ([policy](https://developers.google.com/workspace/workspace-api-user-data-developer-policy)) | Exclude connector tool results from training and from data kept for evals |
| Microsoft 365: admin consent | Users cannot consent to a multitenant app from an unverified publisher if risk-based consent is on ([publisher verification](https://learn.microsoft.com/en-us/entra/identity-platform/publisher-verification-overview)). Since mid-July 2025, third-party access to files and sites will «require\[s\] admin consent» ([MC1097272](https://mc.merill.net/message/MC1097272)) | For M365, become a verified publisher and plan a separate admin consent flow |
| Tokens older than the logs | Gainsight could not find the source of the leak: the tokens were older than the stored logs ([Gainsight](https://www.gainsight.com/blog/how-we-accelerated-a-year-of-security-work-in-weeks/)) | Keep a log of grant issue and use for at least the grant's lifetime |

## iMessage: a channel, not a connector

Talking to an agent in iMessage is a channel, not a tool, so it is not part of AI-816. Sending a message as a tool is possible only through an unofficial API and only on the customer's account. There is one official path: Apple Messages for Business through an Apple-approved provider (MSP). Unofficial iMessage APIs (Sendblue, Linq) work, but Apple does not support them.

Thierry asked on September 30: «also in terms of connectors, what should we do about imessage?» and «yeah lots of growth in that space» ([thread](https://getstream.slack.com/archives/C0BSL73RUDU/p1790784788813469)). All pages below were opened on September 30, 2026.

### Why iMessage is a channel, not a tool

- **An AI-816 connector is a tool.** The model calls it on its own, and Router sends `tools/call` with the connection's token (see «MCP in this document»).
- **iMessage is needed as a channel.** A person writes to the agent in Messages, and the agent answers there. This needs incoming events (a webhook from the provider), linking the chat to a Session, and sending the reply. This is the «Channel or tool» axis from «Evaluation criteria». Channels are out of scope in AI-816 (`connector-design.md:24`).
- **Apple has no iMessage API for arbitrary apps.** A provider says so itself: «No. Apple does not offer a public or official iMessage API» ([Linq](https://linqapp.com/imessage-api)). So the «agent writes to someone in iMessage» case is also possible only through one of the two paths below.

### Two paths

|  | Apple Messages for Business (official) | Unofficial iMessage APIs: Sendblue, Linq |
| --- | --- | --- |
| How it works | A business registers with Apple and connects through an MSP. Apple forwards the customer's messages to the MSP's `/message` endpoint. The MSP replies with `POST https://mspgw.push.apple.com/v1/message` and a JWT in `Authorization` ([MSP REST API](https://register.apple.com/resources/messages/msp-rest-api/)) | The provider gives a dedicated number (line), a REST API for sending and webhooks for incoming messages. If the recipient has no iMessage, it falls back to RCS or SMS ([Sendblue](https://docs.sendblue.com/), [Linq FAQ](https://docs.linqapp.com/guides/resources/faq/)). Linq says it runs its own Macs and Apple accounts (see «Linq: why they are big»). Sendblue's site and docs say nothing about how its lines work inside (checked October 1: not disclosed) |
| Who writes first | The customer: a Messages button on a website, in an app, in Apple Maps, a QR code ([FAQ](https://register.apple.com/resources/messages/messaging-documentation/faq)). Since February 4, 2026 there are Invitations: «Send invitation messages to opted-in customers to initiate a conversation» ([MSP REST API](https://register.apple.com/resources/messages/msp-rest-api/), [Messaging Advisory](https://www.messagingadvisory.com/post/apple-messages-invitations-a-new-chapter-for-apple-messages-for-business)) | Either side. But it is safer if the person writes first: «Recipient-initiated chats can't be reported by the recipient, which reduces the risk of your line being flagged» ([Linq FAQ](https://docs.linqapp.com/guides/resources/faq/)) |
| Rules and limits | Only «Business-to-customer, not person-to-person». A handoff to a live agent is required. Marketing only when the customer asks ([FAQ](https://register.apple.com/resources/messages/messaging-documentation/faq)). A bot «should send the first reply within 5 seconds and clearly identify itself» ([documentation](https://register.apple.com/resources/messages/messaging-documentation/)) | Limits per line. Sendblue on the AI Agent plan: 1000 inbound contacts per day and 200 for follow-up; new group chats cannot be created ([limits](https://docs.sendblue.com/limits/)). Linq: 30 messages per 60 seconds per sender–recipient pair. After complaints a line gets the `FLAGGED` status ([Linq FAQ](https://docs.linqapp.com/guides/resources/faq/)) |
| Risk | Low: Apple built the channel and reviews it (our view) | Apple has shut down unofficial access before. Beeper Mini, December 9, 2023: «We took steps to protect our users by blocking techniques that exploit fake credentials in order to gain access to iMessage» ([AppleInsider](https://appleinsider.com/articles/23/12/10/apple-confirms-it-blocked-beeper-mini-citing-security-risks)) |
| Price | Apple is free; the MSP charges for its services ([FAQ](https://register.apple.com/resources/messages/messaging-documentation/faq)) | Sendblue: $100 per month per line, no per-message fee ([pricing](https://www.sendblue.com/pricing)). Linq: Hobby $0 with 20 contacts, Pro from $260 per month ([Linq](https://linqapp.com/s/pricing)) |

### Linq: why they are big

On October 1 Thierry asked in #video-ai: «like why are these guys big? linqapp.com» ([message](https://getstream.slack.com/archives/C094V4M57NE/p1790865826686939)). Just before that he wrote: «seems like whatsapp, slack and imessage is key to get right» ([message](https://getstream.slack.com/archives/C094V4M57NE/p1790865799094019)). Short answer: Linq was early with a managed iMessage API, AI assistant startups made it their default, and its growth numbers got it a $20M Series A. The growth numbers below are the company's own claims; nobody checked them independently. Pages opened on October 1, 2026.

| Topic | Linq |
| --- | --- |
| What it sells | APIs for agents: «APIs for iMessage, RCS, SMS, and Voice built for Agents» ([linqapp.com](https://linqapp.com)). iMessage is the main product. Messages fall back iMessage → RCS → SMS on the same number ([vs Sendblue](https://linqapp.com/s/linq-vs-sendblue)). Group chats up to 31 handles ([docs](https://docs.linqapp.com/channel/amb/)). Since July 2026 also payments inside the chat: [Agent Pay](https://linqapp.com/blog/introducing-agent-pay-unlocking-commerce-for-imessage-agents) and [Agentcard](https://linqapp.com/blog/your-agent-can-buy-things-now-introducing-agentcard-on-linq) |
| How it delivers iMessage | Linq runs real Macs and Apple accounts: «Linq runs the Macs, Apple accounts, and delivery» ([blog](https://linqapp.com/blog/send-imessage-programmatically)), in «4 redundant data centers» ([vs Photon](https://linqapp.com/s/linq-vs-photon)). Each customer gets its own lines. This is not Apple's official channel: «Linq numbers are P2P iMessage numbers» ([FAQ](https://docs.linqapp.com/channel/imessage/guides/resources/faq/)). Apple Messages for Business is a separate Linq API ([llms.txt](https://docs.linqapp.com/llms.txt)) |
| How a developer connects | REST API with an API key: «There is no OAuth authorization server today» ([auth.md](https://linqapp.com/auth.md)). Keys cannot be limited: «all API keys have full read/write access». Webhooks signed with HMAC-SHA256. SDKs for TypeScript, Python and Go. An MCP server over stdio: `npx -y @linqapp/sdk-mcp@latest` ([FAQ](https://docs.linqapp.com/channel/imessage/guides/resources/faq/)) |
| Agent frameworks | Official adapter for Vercel Chat SDK, June 23, 2026 ([blog](https://linqapp.com/blog/vercel-chat-sdk)). Since August 25, 2026: «Linq is a Vercel Managed Connector» and a `linq` channel in eve ([blog](https://linqapp.com/blog/linq-is-now-a-vercel-managed-connector)) |
| Price | Hobby $0 with 20 contacts, Pro from $260 per month, Enterprise by contract ([pricing](https://linqapp.com/s/pricing)). No per-message fee |
| Funding | Seed $2.5M, May 2021, led by Mucker Capital ([Hypepotamus](https://hypepotamus.com/companies/b2c/with-2-5-million-seed-round-linq-is-building-whats-next-in-networking/)). Series A $20M, February 2, 2026, led by TQ Ventures; valuation not disclosed ([TechCrunch](https://techcrunch.com/2026/02/02/linq-raises-20m-to-enable-ai-assistants-to-live-within-messaging-apps/)). One angel is Matt Fischer, former VP of the App Store at Apple ([linqapp.com/cli](https://linqapp.com/cli)) |
| Growth (company claims) | The API launched in February 2025. In eight months Linq doubled the ARR it had built in four years. Customers grew 132% quarter over quarter; their agents reach 134,000 monthly active users; 30M+ messages a month; net revenue retention 295% with zero churn ([TechCrunch](https://techcrunch.com/2026/02/02/linq-raises-20m-to-enable-ai-assistants-to-live-within-messaging-apps/)). The site now says 60M+ a month ([vs Photon](https://linqapp.com/s/linq-vs-photon)). Customer count does not match: «100+ companies» ([careers](https://linqapp.com/s/careers)) vs «1,000+ world-class teams» ([linqapp.com](https://linqapp.com)) |
| Customers | Poke, Tomo, Hypercard ([Pulse 2.0](https://pulse2.com/linq-20-million-series-a/)); Lindy, Clay, Emergent, Slash, Owner.com ([Linq](https://linqapp.com/s/ai-instructions)) |
| Company | Founded in 2019 in Birmingham, Alabama ([Bham Now](https://bhamnow.com/2026/02/03/birmingham-startup-raises-20m-in-funding-for-ai-powered-messaging/)); founders came from Shipt. It started as a digital business card for sales teams and moved to AI agents after the Poke assistant went viral in September 2025 ([TechCrunch](https://techcrunch.com/2026/02/02/linq-raises-20m-to-enable-ai-assistants-to-live-within-messaging-apps/)) |
| Limits and risks | 30 messages per 60 seconds per sender–recipient pair, under 7,000 messages a day per line ([FAQ](https://docs.linqapp.com/channel/imessage/guides/resources/faq/)), 50 new contacts a day per line ([vs Sendblue](https://linqapp.com/s/linq-vs-sendblue)). A line gets `FLAGGED` after reports or when it looks automated, then is down for about 24 hours ([FAQ](https://docs.linqapp.com/channel/imessage/guides/resources/faq/)). Apple may block it: «There's no telling if Apple will pull a Meta» ([TechCrunch](https://techcrunch.com/2026/02/02/linq-raises-20m-to-enable-ai-assistants-to-live-within-messaging-apps/)) |
| Voice | Not native: «The API does not place or answer voice calls directly»; for voice agents they suggest a VOIP number such as Twilio. Voice memos are supported ([FAQ](https://docs.linqapp.com/channel/imessage/guides/resources/faq/)) |
| WhatsApp and Slack | A WhatsApp API exists in the docs, with Meta's Embedded Signup ([docs](https://docs.linqapp.com/channel/whatsapp/llms.txt)), but the marketing pages do not mention it. Slack is not a channel, only a support channel for customers ([pricing](https://linqapp.com/s/pricing)) |

**Why they are big (our view, from the table above):**

- **They sell the hard part as a service.** Macs, Apple accounts, delivery and line health are Linq's problem. An AI startup gets iMessage through one API key.
- **They were there when iMessage assistants took off.** Poke went viral, AI companies asked for the API, and Linq became their default. That shows in the 295% net revenue retention.
- **They are where developers build agents.** MCP server, CLI, Vercel Chat SDK and a Vercel managed connector.

**What it means for us (our view).** Linq confirms this section: iMessage is a channel, and a real business is built on it. Of Thierry's three channels, Linq really does only iMessage; WhatsApp is in the docs only, Slack is not there. Vercel shows that channels and connectors meet in one place: Linq is both a channel in eve and a connector in Vercel Connect (see «[How Vercel does it: Eve and Connect](#mgrnb5r1kqa.158503)»). For Accelerate Linq is a BYO provider, like Sendblue (BYO is unverified as a decision: asked on October 1 whether the customer brings their own Linq key, Thierry answered «well thats we have to figure out»): its MCP server is stdio, so Router cannot connect it, but its REST API with a static key fits an app-owned connection and the built-in HTTP tool.

Not disclosed (checked October 1 in [TechCrunch](https://techcrunch.com/2026/02/02/linq-raises-20m-to-enable-ai-assistants-to-live-within-messaging-apps/) and the other funding coverage): the valuation («Linq did not disclose its valuation»), ARR in dollars, headcount, how many Macs Linq runs, and whether Apple tolerates this. These stay `unverified` because nobody publishes them, not because they were not looked for.

### Voice agents and Router

- **There is a bridge from a call to Messages.** One of the Invitations entry points is IVR deflection: the caller leaves the queue and continues in Messages ([Messaging Advisory](https://www.messagingadvisory.com/post/apple-messages-invitations-a-new-chapter-for-apple-messages-for-business)). For an Accelerate phone agent this is a natural case (our view). But access is narrow: «Early availability will likely be limited to a small number of brands».
- **iMessage providers have no voice.** «The API does not place or answer voice calls directly» ([Linq FAQ](https://docs.linqapp.com/guides/resources/faq/)). So iMessage is a text Session (`"text": true`), not a voice one.
- **The five voice platforms have no iMessage.** Checked October 1 in each vendor's docs index: the `llms.txt` of [LiveKit](https://docs.livekit.io/llms.txt), [Pipecat](https://docs.pipecat.ai/llms.txt), [ElevenLabs](https://elevenlabs.io/docs/llms.txt) and [PolyAI](https://docs.poly.ai/llms.txt) has no iMessage, Apple Messages for Business or Apple Business Chat. [Retell's index](https://docs.retellai.com/llms.txt) lists phone, web call, chat widget, SMS and the chat API only.
- **Router cannot connect the Sendblue MCP server.** It runs locally with `npx -y sendblue-api-mcp@latest`, and the keys are in environment variables ([MCP](https://docs.sendblue.com/mcp/)). This is stdio, and Router only talks to remote servers over Streamable HTTP (see «MCP in this document»).

### Conclusion (recommendation)

1. **We do not add iMessage to AI-816.** It is a channel, and channels are out of scope. It goes to separate «channels» work together with Slack and WhatsApp (see «Decide separately»).
2. **For Stream's business customers the main path is Apple Messages for Business.** Stream either becomes an MSP or connects to an existing MSP. Apple does document how to become an MSP (checked October 1): register the organization in Apple Business Register, after which an Apple regional team «contacts you to schedule a review of your capabilities and go-to-market plan». Apple expects a messaging platform, a channel connector, a live agent console, bot flows, a customer success team and a conversational designer. An MSP account has three types: Private (development and testing), Commercial Public (listed in the MSP drop-down for new businesses) and Commercial Non-Public ([MSP onboarding](https://register.apple.com/resources/messages/msp-onboarding/), [MSP registration](https://register.apple.com/resources/messages/msp-onboarding/mspRegistration)). No cost or timeline is published.
3. **Unofficial providers: BYO only.** The customer brings their own Sendblue or Linq account and carries the risk of a block. For a «send iMessage» tool, an app-owned connection with a static key and the built-in HTTP tool from the «Work plan» are enough (our view). Stream does not set up its own lines. BYO for Linq is unverified as a decision: on October 1 Thierry answered «well thats we have to figure out», and asked «how do we want to approach this?» for channels in general.

## What this means for Accelerate

The AI-816 data model (connection, owner, binding) is worth keeping: Retell and ElevenLabs reached the same design on their own. The rest (our own OAuth with refresh, the provider catalog, personal connections) is still a hypothesis, tested live on only two providers. From competitors we should take a voice call policy, dependency tracking and a built-in HTTP tool on top of a connection.

Current state of the `codex/connector-support` branch (checked in code on September 28):

- `AgentConnectorBinding` in `acceleration/api/openapi.yaml` has only `name`, `connector_id`, `connection`, `tools[{name, schema_digest}]`, `required` and `timeout_ms` (lines 6162–6198).
- `ConnectorConnection.auth_type` is `oauth2 | none | bearer | api_key`; `status` is `pending | connected | needs_reauthorization | disconnected | failed` (lines 6019–6062).
- `grep -rn client_credentials internal/` finds only `internal/phone/sinch/sinch.go:360`. Connectors have no client credentials grant.
- `internal/api/connectors.go` has no `used_by` and no usages. Nothing tracks which agents use a connection.

### Keep: strong parts of the design, not yet proven in production

| What | Why |
| --- | --- |
| Connections owned by a user | None of the five has end-user OAuth. They all pass tokens through metadata or variables, and the host refreshes them. Athena can't be built without this |
| OAuth per the MCP spec (PKCE, CIMD, DCR) | For MCP, LiveKit, Pipecat, Retell and ElevenLabs offer only headers or client credentials |
| Refresh under a lock with a checkpoint | None of the five documents what happens when a token expires mid-call. Caveat: our integration tests cover this, but live refresh is not checked with any provider |
| Exact grants with `schema_digest` | Matches best practice: `tool_hash` → `needs_review` at ElevenLabs |
| A tool is tied to one connection, not to a provider | Retell (App) and ElevenLabs (`api_integration_connection_id`) work the same way |

### Work plan

One list by priority. P0 is needed before launch: the first stage (app-owned) and Athena on Slack and Linear. P1 comes right after launch, P2 later. Each item has a competitor example or a section with evidence.

**P0: before launch.** Based on [For Athena](#mgrnb5r1kqa.55357), [Incidents and provider quirks](#mgrnb5r1kqa.95263), [Personal token in a voice channel](#mgrnb5r1kqa.75938), [Whose OAuth app](#mgrnb5r1kqa.103491).

1. **A rule for sessions with several people.** Whose connection is used when several people ask the agent in one channel. The safe default is app-owned connections only (see «For Athena» in «Edge cases»). This is Athena's first scenario: «slack connection and multiplayer» (Nash, October 1).
2. **Who owns the OAuth app at the provider.** For Athena: Stream's internal app. The first external customers connect Slack through their own internal app (BYO): until Stream's app is listed in the Marketplace, it can't use Slack MCP, and a review needs at least 10 installs. A shared Stream app comes only after Marketplace review. The API already supports both options; we still need to document this decision. ElevenLabs does the same for Slack and Zendesk.
3. **In a phone session, only app-owned connections by default.** User-owned only after the user is verified outside the voice channel. We don't use device code.
4. **Reliable refresh, tested live on Slack and Linear (a Linear access token lives 24 hours):**
   - don't touch the credential on a temporary error;
   - send `scope`;
   - if the refresh response is lost, retry it within the provider's grace window;
   - write down that the lock works within one DB. The main point: run refresh on its own context, not on the tool call's context. Today an interruption, the 5 s timeout or a network error during refresh moves the connection to `needs_reauthorization` (`internal/connectors/runtime.go:97-112`).

**P1: right after launch.** The first items already exist at competitors (vendor sections and [Comparison](#mgrnb5r1kqa.27281)); the rest come from [Edge cases](#mgrnb5r1kqa.30446) and [Other products](#mgrnb5r1kqa.80042).

1. **Connecting accounts outside the session.** Moved from P0 on October 1: Athena starts with sessions of several people, where only app-owned connections are used.
   - A «connect accounts» page.
   - During a conversation: a one-time link tied to the session and the user. In the callback we check that the same user signs in, like the user verifier at Arcade and session binding at AWS.
   - After consent, the call is retried automatically.
2. **Voice policy on the binding or the tool.**
   - Fields: speech before the call, behavior on interruption, async execution, a «can be cancelled» flag.
   - Examples: `pre_tool_speech`, `interruption_mode`, `execution_mode` at ElevenLabs; `ToolFlag.CANCELLABLE` and `ctx.update()` at LiveKit; `cancel_on_interruption` at Pipecat.
   - Today we have only `timeout_ms`.
3. **A built-in HTTP tool on top of a connection**: for APIs without an MCP server.
   - Examples: the webhook tool with `auth_connection` at ElevenLabs and the APIs tab at PolyAI. Both take auth from storage, not from code.
   - Today such an API can be connected only through caller-executed functions, and then the client holds the credentials.
   - This cuts provider-specific code the most.
4. **Dependency tracking and delete protection.** A list of agents that use a connection, and no delete without `force`. Examples: `list-app-usages` and `force_delete` at Retell, `used_by` at ElevenLabs.
5. **A call log with error types.** Example: ElevenLabs `GET /v1/convai/tools/{id}/executions`: `latency_secs` and `error_type`, split into `customer_auth`, `external_server`, `client_timeout`. The handover already has this as item 7 (telemetry).
6. **OAuth client credentials grant.** Salesforce uses it at Retell and ElevenLabs, and the whole APIs tab at PolyAI uses it. It is the most common case for app-owned service accounts.
7. **Scope check on connect.** Bad example: Retell, where missing scopes go unnoticed. We have `granted_scopes`, but it is only stored and never compared with what the tools need; the connector\_scope\_required error exists only in the design.
8. **Handle `429` and `Retry-After` from the provider.** The connectors code has none of this today.
9. **Tie a connection to the user's lifecycle.** On offboarding, personal connections must be turned off, and on a GDPR request, deleted. Today DELETE is soft: owner\_id and account\_id stay in the row.
10. **Step-up on `insufficient_scope`** during a session: an event for the app, like the one for reconnect today. On step-up, ask for the union of old and new scopes (MCP 2026-07-28).
11. **Revocation signals.** Slack `tokens_revoked` and the Microsoft CAE claim challenge move the connection to `needs_reauthorization`.
12. **Check against MCP 2026-07-28:**
    - `iss` (RFC 9207);
    - bind the client to the `issuer`: today the issuer is stored but not checked on refresh;
    - only `https` endpoints from the metadata: already done, listed for completeness.
13. **Access policy for a Connection:** which agents and users may use it. Similar: permission sets at Salesforce, ACLs at Composio.
14. **`connectors.yaml`:**
    - a required `test` request;
    - templates from the connection config: instance, region, `realmId`;
    - fields from the token response, for example `instance_url`.
15. **Guides for the customer's own app (BYO) for each provider.** The API already takes `oauth_client_id` and `oauth_client_secret` (`api/openapi.yaml:6080-6090`). Missing: step-by-step guides and support for «why doesn't my app work» questions (see «Whose OAuth app»).
16. **Provider review timelines, before we go to external customers.** Slack Marketplace, Google verification and CASA, Microsoft verified publisher. Athena doesn't need them: it runs on Stream's internal app (see «Whose OAuth app»).

**P2: later.**

1. Static egress IPs, a maximum grant age, alerts on call spikes.
2. `read_only` on a binding, and a confirmation for «untrusted content + write».
3. Keep connector results out of training and evals.
4. **Arguments filled in by the backend.** ElevenLabs has them (`constant_value`, `dynamic_variable`), and so does Pipecat (`tools_arguments`). We put `provided_arguments` in stage 4; move it earlier if Athena's first scenario needs it.

### Decide separately

- **Channels are a separate product.** LiveKit uses the word «Connectors» for channels; we use it for tools. Only ElevenLabs has Slack and WhatsApp text channels. iMessage belongs here too (see the «iMessage» section). If Athena needs to «talk from Slack», that is different work: events, mapping a thread to a session, deduplication.
- **Don't chase catalog size.** Providers that are really managed: about 11 at Retell, 20 at ElevenLabs, about 25 documented at PolyAI. MCP and generic HTTP cover the rest. A sensible target is a small tested catalog plus a generic path.
- **Audit.** A call log is observability, not audit. Who issued a grant, who connected an account, who approved a schema change: the connectors code records none of this.
- Token exchange or EMA as a third way to create a user-owned connection. Not available to Stream now: EMA in Claude supports only Okta, and we use Google Workspace.
- Do we need a «host brings the token» path, like the Vapi override, for customers with their own vault?

### Risks of our approach (assessment)

1. **We take on storing other people's tokens.**
   - A DB leak together with the KEK gives access to every connected customer account.
   - Migrating from the old plugins can't be undone, but it seems not to be needed: Router has no production deployment, only staging (see «Do we need the move at all» in «Connectors and how they differ from plugins») (`connector-handover.md`).
   - LiveKit and Pipecat avoid this responsibility on purpose; brokers build a whole business on it.
2. **We may build for ourselves, not for customers.**
   - Personal connections are needed by Athena, an internal product.
   - There are no customers yet. If the target market is contact centers and doctor appointments, app-owned is probably enough.
   - Athena is dogfooding. Its needs show what an «employee assistant» product needs, but they don't prove this is our target market. The target market is not set; the decision is ours. The guard against this risk is the work order in the TL;DR.
3. **Latency in voice.**
   - Credentials are resolved on every request, and refresh runs under a Postgres lock, all in the call path.
   - Not measured (`connector-handover.md`, remaining work item 8).
   - Competitors that set tokens up front in headers don't pay this cost.
4. **Friction from `schema_digest`.**
   - Any edit to a tool description on a hosted MCP server (Slack, Linear) hides the tool until review.
   - For a server that updates without notice, this is a silent loss of a feature in production.
   - ElevenLabs does something similar, but with an approval UI and a `needs_review` status. We haven't measured how often Slack and Linear change their descriptions.
5. **Amount of work.**
   - Before release: 9 remaining work items, plus the items in this document.
   - The connectors plan is one week ([Thierry's estimate](https://getstream.slack.com/archives/C094V4M57NE/p1790528558652349), September 27).
   - Narrowing the release to the first stage from the TL;DR (app-owned) plus Slack and Linear for Athena is more realistic than finishing all 7 providers.

## Open questions

### Product questions (we decide ourselves, checking with Nash)

- [x] **Which scenario do we support: agents for businesses or employee assistants?** Decided on September 29: Thierry left the decision to us, and there is no clear requirement. We support both, and the developer picks the owner. Still open: do we build our own catalog of personal providers or plug in a broker.
- [ ] What does Athena need first: Slack as a tool (read and write), Slack as a channel, or both?
- [ ] Should we build a built-in HTTP tool on top of a connection before growing the MCP provider catalog?
- [ ] Do the first app-owned service accounts need the client credentials grant?
- [ ] Where do we store the voice call policy: in the binding or in each tool grant?
- [ ] How do we approve calls on the phone, where there is no UI? ElevenLabs hasn't solved this either.
- [ ] Which rule for sessions with several people do we pick for Athena in a Slack channel?
- [ ] Slack for Stream customers: do we go through Marketplace review, or require the customer's own app (BYO)?
- [ ] Do we plug in a broker for some providers? We decide on data after a month of our own layer running on Slack and Linear (see «Build our own layer or use a broker»).
- [ ] Which providers do the first external customers need? This decides which reviews we must pass.
- [ ] Do we need white-label: our customer's domain in the redirect URI and their brand on the consent screen?
- [ ] Do we let BYO customers export their tokens, who and how? For: it is a fair way out, and security reviews ask about it. Against: an endpoint that returns secrets is a risk by itself and needs strict permissions and audit. Option: export on request through support, not through a public API.
- [ ] Who at Stream owns the OAuth apps at providers: developer accounts, secrets, their rotation?
- [ ] Do customers need iMessage as a channel? If yes, do we become an MSP or go through an existing one? Apple's public docs don't say how to become an MSP.

### Need people or accounts

- [ ] **Live vendor accounts and latency measurements.** We need accounts at Retell, ElevenLabs and PolyAI, and provider keys. We chose not to sign up on Stream's behalf.
- [ ] **Behavior inside closed platforms.** This covers MCP headers at Retell, `revoked` at ElevenLabs and encryption at PolyAI. It can only be seen on a live account.
- [ ] **Sierra, Decagon, Parloa.** We need access to their docs.
- [ ] **OpenAI Connector Registry.** The docs page was removed.
- [ ] **Can we use Nango's catalog data?** The providers.yaml format can serve as a model; whether we can copy the data itself is a question for a lawyer.
- [ ] **Linear ticket AI-816.** It names the wrong Volt branch and needs a fix.

## Appendix: fact check (September 29)

This is a check of the doc's disputed and unverified claims, done on September 29, with the rows marked «Re-checked October 1» refreshed on that date. If the text above disagrees with it, this table is right. Sources: vendor docs and OpenAPI, source code, GitHub issues and Stream's Slack. Three items were checked by experiment: a 401 from an MCP server in Pipecat, a shared `MCPServer` in LiveKit, and the AI-816 tests on local Postgres and Redis.

Statuses:

- «Confirmed»: what the doc says is right.
- «Fixed»: it was wrong and is already fixed in the text above.
- «Partial»: only part of the question has an answer.
- «Could not check»: reasons at the end of the section.

**Accelerate (AI-816)**

| Item | Status | What we found | Source |
| --- | --- | --- | --- |
| Interruption during an MCP call | Confirmed | The call is cancelled, the MCP server gets `notifications/cancelled`, and the error text goes into the history. There is no separate test for this | `internal/agent/agent.go:1182-1188`, `internal/mcp/mcp.go:300` |
| Deleting a user's connections (GDPR) | Confirmed | Not there. `DELETE` is soft: `owner_id` and `account_id` stay in the row, and there is no foreign key to the user | `internal/store/connectors.go:294-306`, `migrations/20260929120000_connectors.sql:12` |
| Branch mismatch between Linear and the handover | Fixed | The Linear ticket was wrong. The backend lives in `codex/connector-support`, and `codex/connector-support-handover` is the Volt branch. `e8faace9` and `38f7869c` are normal `accelerate` commits | `connector-handover.md:10-16`, `git branch -a --contains` |
| `reconnect_required` and the timeout in the design doc | Confirmed | Line 383 is out of date: the code uses `needs_reauthorization`, and there is also a `failed` status. Line 64 is technically right but misleading: each binding has its own timeout, 5 s by default | `connector-design.md:64,383`, `internal/session/connector_tools.go:19,146-149` |
| MCP 2026-07-28 requirements | Partial | `iss` is checked if the server says it supports it; endpoints are https only. No union of scopes on step-up; binding the client to `issuer` is partial | `internal/mcp/oauth.go:522-529,539-595`, `internal/egress/public.go:48` |
| Refresh: `scope` and temporary errors | Fixed | `scope` is not sent. A network error, an interruption or the 5 s timeout during refresh leave the connection in `needs_reauthorization`; the credentials are kept | `internal/mcp/oauth.go:543-548`, `internal/connectors/runtime.go:97-112`, `internal/api/integration_test.go:861-930` |
| Scope check, 429, usage tracking | Confirmed | None of this exists: `connector_scope_required` is never returned, 429 and `Retry-After` are not handled, there is no `used_by` | `internal/store/models.go:374-377`, `internal/mcp/mcp.go:202` |
| Refresh lock and tests | Confirmed | A session `pg_advisory_lock` on the tenant + connection pair, and an atomic write by `revision`. The race and lost-response tests pass; no live provider in the tests | `internal/store/connectors.go:260-268,323-346`; `go test -tags integration ./internal/api` 29.09 |

**LiveKit and Pipecat**

| Item | Status | What we found | Source |
| --- | --- | --- | --- |
| LiveKit Connectors is a closed service | Confirmed | «available in LiveKit Cloud only». Only the protobuf definitions and the SDK clients are open | [twilio](https://docs.livekit.io/telephony/connectors/twilio.md), `livekit/protocol` `livekit_connector.proto:25-34` |
| LiveKit: one `MCPServer` for several sessions | Confirmed | Not possible. Closing one session breaks the connection for all. In the experiment, the second session got «internal service is unavailable» | `mcp.py:115-128,295-302` @ `57b3227a` |
| LiveKit: MCP in Node | Fixed | agents-js has no MCP client, although the README promises «Native support for MCP». PR #2492 with an implementation is open | `@livekit/agents` 1.9.1, [MCP](https://docs.livekit.io/agents/logic/tools/mcp.md) |
| Pipecat: expired MCP token | Confirmed | Experiment: the 401 becomes the string «Server returned an error response» for the model. The session is not reset, and there is no refresh | `mcp_service.py:50-62,106-107,574` @ `94e40901` |
| Pipecat `COMMUNITY_INTEGRATIONS.md` | Fixed | It is a guide for integration authors, not a list. It says nothing about OAuth. The list itself is 65 Community rows in the docs | `COMMUNITY_INTEGRATIONS.md:311`, `pipecat-ai/docs` `supported-services.mdx` |
| Tool rate limits | Confirmed | Neither documents them. LiveKit only has `max_tool_steps=3` per turn | `agent_session.py:402` |

**Voice platforms**

| Item | Status | What we found | Source |
| --- | --- | --- | --- |
| Retell: `connect-app` and `list-app-templates` | Fixed | These are not endpoints: the names appear only in the descriptions of other methods, and the docs pages return 404 | `src/retell/resources/app.py:70,75,138`, `openapi-final.yaml:8755,8807` |
| Retell: refresh for Zendesk, Salesforce, Dynamics | Confirmed | All of them need a manual Reconnect after an error. Salesforce uses client credentials and has no refresh tokens; Dynamics stores a refresh token | [Zendesk](https://docs.retellai.com/integrations/zendesk), [Salesforce](https://docs.retellai.com/integrations/salesforce), [Dynamics](https://docs.retellai.com/integrations/microsoft-dynamics) |
| Retell: headers after `update-live-call` | Partial | In a custom function, the value is read when it is used. In MCP, variables are filled in «when it connects»; the docs don't say if they are re-read during a call | [dynamic variables](https://docs.retellai.com/build/dynamic-variables), [MCP](https://docs.retellai.com/build/single-multi-prompt/mcp) |
| Retell: interruption during a call | Confirmed | A custom function «runs to completion (up to the timeout)» | [custom function](https://docs.retellai.com/build/single-multi-prompt/custom-function) |
| ElevenLabs: OAuth per the MCP spec | Fixed | Not supported. There are only a secret token, headers and auth connections: client credentials, JWT, basic, bearer, mTLS | `openapi.json` `MCPServerConfig-Input`, [changelog](https://elevenlabs.io/docs/changelog) |
| ElevenLabs: `revoked` mid-conversation | Could not check | The token manager writes the status. Nothing describes what the caller hears then | [auth connections](https://elevenlabs.io/docs/api-reference/workspace/auth-connections/list.md) |
| ElevenLabs: webhook tool retries | Fixed | No retries. `retry_enabled` exists only for workspace event webhooks | `openapi.json` `WebhookToolConfig-Input` |
| ElevenLabs: several connections to one provider | Confirmed | Possible: the connection is set on each tool | `api_integration_connection_id`, `auth_connection` in `openapi.json` |
| ElevenLabs: 20 native integrations | Confirmed | Four use OAuth through the ElevenLabs app. The rest use keys, basic or client credentials | [llms.txt](https://elevenlabs.io/docs/llms.txt) |
| PolyAI: Connect Portal and «130+ integrations» | Confirmed | Marketing only. The docs have nothing like it; there are about 19 integrations | [integrations](https://poly.ai/integrations) |
| PolyAI: OAuth other than client credentials | Partial | The APIs tab and MCP have only client credentials. The Salesforce integration takes the integration user's login and password | [Salesforce](https://docs.poly.ai/integrations/salesforce) |
| PolyAI: secret encryption and rotation | Partial | PolyAI doesn't track rotation: «does not track or enforce rotation cadence». Encryption gets one sentence | [update a secret](https://docs.poly.ai/api-reference/agents/endpoint/secrets/update-a-secret) |
| PolyAI: caller identity check | Confirmed | These are flows the customer writes (`conv.state.is_verified`). There is no ready-made engine | [triggering flows](https://docs.poly.ai/flows/triggering-flows) |
| Tool rate limits at Retell, ElevenLabs, PolyAI | Confirmed | None documents them; there are only timeouts | vendor sections above |

**Slack and brokers**

| Item | Status | What we found | Source |
| --- | --- | --- | --- |
| Slack limits for MCP | Confirmed | For MCP «the same rate limits apply» as for the Web API. Apps that are not in the Marketplace and not internal can't use MCP at all | [Slack MCP](https://docs.slack.dev/ai/slack-mcp-server/) |
| Slack Marketplace review threshold | Confirmed | At least 10 installs in active workspaces | [changelog](https://docs.slack.dev/changelog/2026/09/01/slack-marketplace-install-requirement/) |
| Pipedream and Paragon prices | Partial | Pipedream: $99 a month billed yearly, 100 external users, then $2 each. Paragon has no public prices | [Pipedream](https://pipedream.com/pricing), [Paragon](https://www.useparagon.com/pricing) |
| Vercel Connect price | Fixed | $3 per 1,000 token requests. «$3 per 10,000» is the old price, valid until 25.09.2026 | [pricing](https://vercel.com/docs/connect/pricing) |
| Vercel Connect and Slack MCP | Could not check | Re-checked October 1: the Vercel catalog lists Slack only as a «Managed» connector, a Vercel-developed Slack app installed per workspace, and no entry names mcp.slack.com. A Custom OAuth connector can point at any OAuth MCP URL with the customer's own client, so Slack MCP through Connect would need the customer's own internal or Marketplace app. Not tested live (our view) | [Connect](https://vercel.com/docs/connect), [catalog](https://vercel.com/connect/browse) |
| How providers count rate limits | Confirmed | Google and Microsoft Graph have one shared cap for all customers of one app. Slack, Salesforce, HubSpot and Linear don't | [Graph](https://learn.microsoft.com/en-us/graph/throttling-limits), [Calendar quota](https://developers.google.com/workspace/calendar/api/guides/quota) |

**Assistants and IDEs**

| Item | Status | What we found | Source |
| --- | --- | --- | --- |
| Cursor | Fixed | DCR or a static client, no CIMD. OAuth per user, including for team servers | [cloud agents](https://cursor.com/docs/cloud-agent/capabilities) |
| VS Code: where tokens are stored | Confirmed | In SecretStorage; the encryption key is in the OS keychain | `dynamicAuthenticationProviderStorageService.ts:184-187` @ `f20366e4` |
| OpenAI Connector Registry | Could not check | Re-checked October 1: openai.com and help.openai.com answer automated reads with a bot challenge. Search snippets of OpenAI's AgentKit post say the registry «is rolling out to ChatGPT Enterprise, ChatGPT Edu, and API users through the Global Admin Console», and a June 3, 2026 update winds down Agent Builder and Evals on November 30, 2026. Who stores the credentials is still unknown | [AgentKit](https://openai.com/index/introducing-agentkit/) |
| Dust: personal credential not connected yet | Confirmed | The agent is paused, and the user sees a «Connect account» card | `front/lib/actions/mcp_internal_actions/events.ts:67-73` @ `53637106` |
| ChatGPT: token expired | Partial | ChatGPT tries a refresh. On `invalid_grant`: «Reconnect required». The docs don't describe this | [codex#47513](https://github.com/openai/codex/issues/47513) |
| Slack MCP: token lifetime | Partial | Not in the docs. The server metadata on 29.09 lists the `refresh_token` grant | `mcp.slack.com/.well-known/oauth-authorization-server` |
| Claude: when it refreshes a token | Confirmed | On 401 and «up to five minutes before the stored expiry» | [authentication](https://claude.com/docs/connectors/building/authentication) |

**Providers and other**

| Item | Status | What we found | Source |
| --- | --- | --- | --- |
| Stream's IdP for EMA | Confirmed | Employees use Google Workspace, and EMA in Claude supports only Okta for now | #internal-it; [blog](https://claude.com/blog/enterprise-managed-auth) |
| Google RISC and refresh tokens | Partial | `token-revoked` applies to refresh tokens, `sessions-revoked` doesn't | [RISC](https://developers.google.com/identity/protocols/risc) |
| Notion: token lifetime | Fixed | 8 hours and 180 days apply only to Notion MCP. REST OAuth has no expiry, and the refresh token changes on every refresh | [MCP client](https://developers.notion.com/guides/mcp/build-mcp-client), [authorization](https://developers.notion.com/guides/get-started/authorization) |
| HubSpot: refresh tokens | Confirmed | An access token lives 30 minutes. Refresh tokens «don't expire (for now)», but a refresh may return a new one | [tokens](https://developers.hubspot.com/docs/api-reference/legacy/authentication/manage-oauth-tokens.md) |
| Salesforce: 5 approvals | Confirmed | «After a fifth approval is made, the oldest approval is revoked» | [help](https://help.salesforce.com/s/articleView?id=xcloud.remoteaccess_request_manage.htm&type=5) |
| Nango catalog license | Confirmed | No separate license; the ELv2 of the whole repo applies | npm `@nangohq/providers` 0.71.10 |
| Sierra, Decagon, Parloa | Could not check | Re-checked October 1: docs.parloa.com has a public landing page, but its own answer engine says the detailed spaces «require a separate login and whitelisting»; Decagon and Sierra publish no developer docs. An archived copy of Parloa shows only basic and bearer | [Decagon security](https://decagon.ai/security) |
| AWS AgentCore and the phone | Confirmed | Nothing about consent without a browser; 3LO is a browser redirect | [session binding](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/oauth2-authorization-url-session-binding.html) |
| Salesforce: agent user and personal auth | Fixed | Agent user and `VerifiedCustomerId` are confirmed. For MCP in Agentforce «User-level authentication isn't supported», and a Salesforce blog says the opposite | [considerations](https://help.salesforce.com/s/articleView?id=ai.agent_mcp_considerations.htm&language=en_US&type=5), [Voice guide](https://github.com/salesforce/einstein-platform/blob/main/resources/afv-implementation-guide/03-build.md) |
| SMS link for OAuth during a call | Could not check | Re-checked October 1: still no vendor offers it as a product feature. [ElevenLabs' guide to caller authentication](https://elevenlabs.io/blog/designing-secure-caller-identity-authentication-flows-for-voice-agents) lists five methods and none sends a link; its one-time code is read back to the agent. An [AWS reference design by SmartBots](https://www.smartbots.ai/securing-voice-based-transactions-using-oauth2-0-and-multimodality/) (Amazon Connect, Lex, Cognito, Twilio SMS) sends a scoped Cognito link during the call, built by hand. The closest product feature is Call companion in Dialogflow CX | [call companion](https://docs.cloud.google.com/dialogflow/cx/docs/concept/call-companion) |

Side effect of the check: the AI-816 integration tests recreated the schema of the local `model_router_test` database in the `vision-agents-postgres-1` container.

## Sources

All pages were opened on September 28, 2026. Below are the key ones; other links sit next to the claims in the vendor sections.

Pages opened on September 29 are cited right in the text, next to each fact. These are the sections «Other products», «Whose OAuth app», «Personal token in a voice channel», «Incidents and provider quirks» and «Appendix: fact check».

**LiveKit**

- [Connectors overview](https://docs.livekit.io/telephony/connectors/)
- [WhatsApp Connector](https://docs.livekit.io/telephony/connectors/whatsapp.md)
- [Twilio Connector](https://docs.livekit.io/telephony/connectors/twilio.md)
- [Introducing LiveKit Connectors](https://livekit.com/blog/introducing-livekit-connectors)
- [MCP](https://docs.livekit.io/agents/logic/tools/mcp/)
- [Async tools](https://docs.livekit.io/agents/logic/tools/async/)
- [Secrets](https://docs.livekit.io/deploy/agents/secrets/)
- [External data](https://docs.livekit.io/agents/logic/external-data.md)
- [livekit/agents@57b3227a](https://github.com/livekit/agents/tree/57b3227a7842697e6ad45b1275369cf9700bf161)
- [livekit/protocol@26ce7b82](https://github.com/livekit/protocol/tree/26ce7b82f4011a6d77058e3bc1587e7117315a9a)

**Pipecat**

- [Supported Services](https://docs.pipecat.ai/api-reference/server/services/supported-services.md)
- [MCPClient](https://docs.pipecat.ai/server/utilities/mcp/mcp)
- [Function calling](https://docs.pipecat.ai/guides/learn/function-calling)
- [Pipecat Cloud Secrets](https://docs.pipecat.ai/pipecat-cloud/fundamentals/secrets.md)
- [WhatsApp](https://docs.pipecat.ai/pipecat/features/whatsapp.md)
- [Telephony in production](https://docs.pipecat.ai/pipecat/deployment/telephony-in-production.md)
- [pipecat@94e4090](https://github.com/pipecat-ai/pipecat/tree/94e40901655e9e082b44c15059c0b3e65b856f37)
- [pipecat-examples@93c0bd5](https://github.com/pipecat-ai/pipecat-examples/tree/93c0bd5f31e18d5d182a49f4a5eeead60b640db6)

**Retell AI**

- [Integrations catalog](https://www.retellai.com/integrations)
- [Integrations overview](https://docs.retellai.com/integrations/overview.md)
- [Integration tools](https://docs.retellai.com/build/single-multi-prompt/integration-tools.md)
- [Custom function](https://docs.retellai.com/build/single-multi-prompt/custom-function.md)
- [MCP](https://docs.retellai.com/build/single-multi-prompt/mcp.md)
- [Dynamic variables](https://docs.retellai.com/build/dynamic-variables.md)
- [Secure webhook](https://docs.retellai.com/features/secure-webhook.md)
- [retell-python-sdk](https://github.com/RetellAI/retell-python-sdk) @ `75a6a5a`

**ElevenLabs Agents**

- [Tools](https://elevenlabs.io/docs/eleven-agents/customization/tools)
- [Webhook tools](https://elevenlabs.io/docs/eleven-agents/customization/tools/webhook-tools)
- [MCP](https://elevenlabs.io/docs/eleven-agents/customization/tools/mcp)
- [Dynamic variables](https://elevenlabs.io/docs/eleven-agents/customization/personalization/dynamic-variables)
- [Environment variables](https://elevenlabs.io/docs/eleven-agents/integrate/environment-variables)
- [Zendesk](https://elevenlabs.io/docs/eleven-agents/customization/integrations/zendesk)
- [Slack](https://elevenlabs.io/docs/eleven-agents/customization/integrations/slack)
- [WhatsApp](https://elevenlabs.io/docs/eleven-agents/whatsapp)
- [OpenAPI](https://api.elevenlabs.io/openapi.json)
- [Agent AsyncAPI](https://github.com/elevenlabs/packages/blob/main/packages/types/schemas/agent.asyncapi.yaml)

**PolyAI**

- [Integrations](https://docs.poly.ai/integrations/introduction)
- [API integrations](https://docs.poly.ai/integrations/api/introduction)
- [ADK api\_integrations](https://polyai.github.io/adk/reference/resources/api_integrations/)
- [Secrets](https://docs.poly.ai/secrets/introduction.md)
- [MCP](https://docs.poly.ai/mcp/agent-studio-integrations.md)
- [Managed services](https://docs.poly.ai/integrations/managed-services.md)
- [Salesforce](https://docs.poly.ai/integrations/salesforce.md)
- [Handoffs](https://docs.poly.ai/voice-channel/handoffs.md)
- [poly.ai/integrations](https://poly.ai/integrations) (marketing)

**Accelerate**

- Branch [Vision-Agents/codex/connector-support](https://github.com/GetStream/Vision-Agents/tree/codex/connector-support):
  - `acceleration/docs/connector-design.md`
  - `acceleration/docs/connector-handover.md`
  - `acceleration/api/openapi.yaml`
- [AI-816](https://linear.app/stream/issue/AI-816/basic-connectorsmcp-support)

**Auth brokers**

- [Vercel Connect](https://vercel.com/docs/connect), [tokens](https://vercel.com/docs/connect/concepts/tokens), [pricing](https://vercel.com/docs/connect/pricing)
- [eve-connectors-research.md](https://github.com/GetStream/Vision-Agents/blob/codex/connector-support/acceleration/docs/eve-connectors-research.md): our review of Vercel Eve and Connect
- [Composio authentication](https://docs.composio.dev/docs/authentication), [pricing](https://composio.dev/pricing)
- [Nango token refreshing](https://nango.dev/docs/guides/auth/token-refreshing), [webhooks](https://nango.dev/docs/guides/platform/webhooks-from-nango), [pricing](https://nango.dev/pricing)
- [Arcade auth providers](https://docs.arcade.dev/en/references/auth-providers), [pricing](https://arcade.dev/pricing)
- [Pipedream Connect](https://pipedream.com/docs/connect)
- [Paragon](https://docs.useparagon.com)
- [Merge Agent Handler](https://docs.merge.dev/merge-agent-handler/how-it-works), [pricing](https://merge.dev/pricing/agent-handler)

**LLM vendors**

- [Claude connectors: authentication](https://claude.com/docs/connectors/building/authentication)
- [Anthropic MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector), [Managed Agents vaults](https://platform.claude.com/docs/en/managed-agents/vaults)
- [OpenAI Responses MCP](https://developers.openai.com/api/docs/guides/tools-connectors-mcp), [ChatGPT apps auth](https://developers.openai.com/plugins/build/auth)

**Specs and providers**

- MCP 2026-07-28: [authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization), [tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools), [changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog), [security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices)
- [RFC 9700](https://www.rfc-editor.org/rfc/rfc9700) (OAuth 2.0 Security BCP)
- Slack: [MCP server](https://docs.slack.dev/ai/slack-mcp-server/), [rate limits](https://docs.slack.dev/apis/web-api/rate-limits/), [token rotation](https://docs.slack.dev/authentication/using-token-rotation/), [Events API](https://docs.slack.dev/apis/events-api/), [org-ready apps](https://docs.slack.dev/enterprise/organization-ready-apps/)
- [Linear OAuth 2.0](https://linear.app/developers/oauth-2-0-authentication)
- Google: [OAuth 2.0](https://developers.google.com/identity/protocols/oauth2), [restricted scope verification](https://developers.google.com/identity/protocols/oauth2/production-readiness/restricted-scope-verification)
- [WhatsApp: send messages](https://developers.facebook.com/documentation/business-messaging/whatsapp/messages/send-messages)

**MCP (section «What MCP is», opened September 29)**

- [Intro](https://modelcontextprotocol.io/docs/getting-started/intro), [architecture](https://modelcontextprotocol.io/docs/learn/architecture), [server concepts](https://modelcontextprotocol.io/docs/learn/server-concepts), [client concepts](https://modelcontextprotocol.io/docs/learn/client-concepts)
- Spec 2026-07-28: [basic](https://modelcontextprotocol.io/specification/2026-07-28/basic), [transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports), [stdio](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio), [Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http), [resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources), [prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts), [security considerations](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/security-considerations)
- [Security best practices](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)

**iMessage (opened September 30)**

- Apple Messages for Business: [FAQ](https://register.apple.com/resources/messages/messaging-documentation/faq), [docs](https://register.apple.com/resources/messages/messaging-documentation/), [MSP REST API](https://register.apple.com/resources/messages/msp-rest-api/)
- [Messaging Advisory: Invitations](https://www.messagingadvisory.com/post/apple-messages-invitations-a-new-chapter-for-apple-messages-for-business)
- Sendblue: [docs](https://docs.sendblue.com/), [limits](https://docs.sendblue.com/limits/), [pricing](https://www.sendblue.com/pricing), [MCP](https://docs.sendblue.com/mcp/)
- Linq: [iMessage API](https://linqapp.com/imessage-api), [FAQ](https://docs.linqapp.com/guides/resources/faq/)
- Linq company and product: [pricing](https://linqapp.com/s/pricing), [iMessage FAQ](https://docs.linqapp.com/channel/imessage/guides/resources/faq/), [auth.md](https://linqapp.com/auth.md), [send iMessage programmatically](https://linqapp.com/blog/send-imessage-programmatically), [Vercel Chat SDK](https://linqapp.com/blog/vercel-chat-sdk), [Vercel Managed Connector](https://linqapp.com/blog/linq-is-now-a-vercel-managed-connector)
- Linq funding and growth: [TechCrunch, Series A](https://techcrunch.com/2026/02/02/linq-raises-20m-to-enable-ai-assistants-to-live-within-messaging-apps/), [Hypepotamus, seed](https://hypepotamus.com/companies/b2c/with-2-5-million-seed-round-linq-is-building-whats-next-in-networking/), [Bham Now](https://bhamnow.com/2026/02/03/birmingham-startup-raises-20m-in-funding-for-ai-powered-messaging/), [Pulse 2.0](https://pulse2.com/linq-20-million-series-a/)
- [AppleInsider: Beeper Mini](https://appleinsider.com/articles/23/12/10/apple-confirms-it-blocked-beeper-mini-citing-security-risks)

**Other voice platforms**

- [Vapi MCP](https://docs.vapi.ai/tools/mcp), [Vapi server authentication](https://docs.vapi.ai/server-url/server-authentication), [Bland tools](https://docs.bland.ai/agents/tools)

  |  |  |  |
  | --- | --- | --- |
  |  |  |  |
  |  |  |  |
