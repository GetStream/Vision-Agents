# AGENTS.md

When running an agent using @acceleration/ or an example using @acceleration/ refer to the skill acceleration/development_skills/dashboard_skill.md to setup the dashboard and launch the example. Make sure you restart any routers or binaries to avoid running old builds.

Work happens on the `accelerate` branch here, and the agents docs live in getstream.io's `content/docs` (see the `docs` skill).

## Engineering integrity

These are acceptance criteria for every change, including code, prompts, agent configs,
tests, benchmarks and reports. A passing test or a higher score does not justify a change
that violates them. Apply them together with the relevant directory instructions and skills.

### Benchmark neutrality

Improve the behavior a real user receives. The implementation must not gain an advantage
from knowing it is being evaluated. The same documented inputs and configuration must use
the same production behavior inside and outside a benchmark.

- **No benchmark detection.** Do not select a favorable execution path from scenario names,
  fixture IDs, evaluation flags, call metadata, known transcripts, seeds or the runner's
  environment. Do not move that detection into a prompt, config, helper or dependency.
- **No answer leakage.** The system under evaluation must not consume expected answers,
  future caller turns, reference transcripts, expected tool sequences, judge instructions or
  hidden world state through its code or prompts. Obtain facts through the same user input
  and public tools available in real use. A published task contract is legitimate input;
  a scenario's answer key is not.
- **No rehearsed solutions.** Do not hardcode benchmark phrases, entities, tool arguments,
  response sequences, pauses or thresholds to satisfy known cases. A rule or fast path needs
  a general reason grounded in the product contract, protocol or measured workload, plus
  evidence on inputs beyond the cases that motivated it. Provider-specific behavior is
  justified by that provider's documented semantics, never by its benchmark result.
- **No cheaper semantics.** Do not skip authentication, validation, persistence, accounting,
  cancellation, cleanup or other required work to make a run faster. Do not count queued
  work as completed, a filler as a substantive answer, or dropped work as a latency win.
  Performance improvements must preserve correctness and the promised durability and quality.
- **No moving the goalposts.** Do not weaken assertions, tolerances, timeouts, workload,
  quality gates, judge prompts or failure classification to make a candidate pass. Do not
  remove difficult cases, hide failed trials, retry until green or report only favorable runs.
  An actual evaluation bug needs its own reproducer and an independently reviewable fix;
  re-evaluate both baseline and candidate with the same corrected methodology.
- **No favored target.** Give compared systems the same task information and apply the same
  scoring rules. Match models, tools, resources, warmup, caches and retry budgets for a claim
  about implementation efficiency. Disclose intentional differences as a comparison of those
  configurations; do not attribute their effect to a code change.

Fixtures, deterministic seeds, cached caller audio and explicit mock providers belong in
test or benchmark setup and may isolate a stated cost. They must not supply hidden hints
or shortcuts to the shared production path. In particular, `performance_benchmark/` measures
the router with `mock/*` providers: exploiting their fixed text, tone or timing is not a
router optimization, and its results do not establish live-provider quality.

### Keep code necessary

- Smallest possible diff. Prefer deleting code over adding it.
- Every added layer, option, dependency, check and log must serve the stated requirement or
  a demonstrated correctness need. Do not add speculative compatibility, generic frameworks
  for one caller, or "future-proofing". Validate at trust boundaries; do not scatter redundant
  checks around internal invariants instead of fixing the code that broke them.
- Reuse the existing owner of a behavior. Avoid parallel implementations and redundant
  wrappers. New caches, retries and fallbacks must solve a demonstrated requirement.
- Match the style and abstraction level of surrounding code. Keep unrelated refactors,
  formatting changes and generated churn out of the diff. Do not remove valid comments.
- Fix root causes. Do not swallow exceptions, fabricate successful results, silently return
  defaults, or add sleeps/retries to hide races. Handle expected failures at the boundary
  that owns them, with bounded work and observable failure when recovery is exhausted.
- Keep resource ownership explicit. Background work needs an owner and a defined lifetime,
  cancellation and shutdown behavior; queues, retries and caches need bounds, and caches
  need correct invalidation and tenant isolation. Moving work off the hot path must preserve
  its completion and failure guarantees.
- Comments explain intent, constraints or a non-obvious tradeoff. Avoid narration of the code,
  unsupported claims and placeholder TODOs. Remove dead paths, unused config, debug output
  and abandoned experiments before delivery.

### Tests must catch a defect

Before adding a test, name the incorrect behavior it would catch. If a plausible broken
implementation would still pass, strengthen the test or leave it out. Coverage and test count
are not evidence of correctness.

- Assert the contract's observable output, state change or failure. "Did not crash", non-null,
  non-empty or HTTP 200 alone is insufficient when the requirement concerns returned data,
  persistence, ordering or side effects. Check the part that could be wrong.
- Do not test your own setup: a mock returning its configured value, a constructor echoing
  its arguments, or an assignment followed by an equality check proves no product behavior.
  Checking that a helper was called is not proof that its promised effect happened.
- Keep expected results independent. Do not copy the implementation's algorithm into the
  assertion, regenerate a golden from the candidate and accept it blindly, or inspect source
  text for tokens instead of executing the behavior.
- For a regression, show the test failing for the original defect before the fix where
  practical. Otherwise explain why and identify the wrong result it rejects. A deliberate
  local perturbation can check sensitivity; do not add a mutation-testing framework just to
  satisfy this rule.
- Cover relevant boundaries and negative cases: changed values, input sizes, event order,
  cancellation and errors. A fix for one transcript needs varied phrasing, not just its
  literal words. Use table-driven cases when each exercises a distinct risk, not many copies
  of the happy path. For asynchronous behavior, wait for observable state with a deadline;
  sleeping and hoping is not synchronization.
- Do not add tests solely for comments, formatting, trivial forwarding or implementation
  details with no behavioral contract. Use existing checks for mechanical changes. Do not
  weaken or delete a meaningful test because new code fails it; justify a contract change
  explicitly. Test code must meet the same necessity and readability bar as production code.

### Evidence before claims

Before changing behavior, state the user-visible failure or cost, the invariant to preserve,
and why the proposed mechanism should help. Then validate that explanation:

- For an improvement claim, keep validation cases out of tuning. Record what was used to
  develop the fix and what was reserved for validation; once inspected and tuned against,
  a case is no longer held out. Do not call a known scenario pool an unseen evaluation.
- Define the metric, workload, repetitions and treatment of failures before comparing.
  Record base and candidate commits, build/run commands, harness revision, seeds, configs,
  models and relevant hardware/network conditions. Rebuild what is measured. Keep these
  conditions fixed except for the declared change; follow the benchmark skill's repeat and
  noise requirements. A microbenchmark improvement alone is not an end-to-end improvement.
- Preserve the raw artifacts for the comparison, including unsuccessful runs, and link them
  without secrets or private caller data. Report sample counts, variation, exclusions and
  regressions alongside gains. Separate infrastructure failures with evidence and keep them
  visible; a candidate's crash, timeout or dropped work is not disposable infrastructure noise.
- Say when a result is inconclusive, untested or blocked. Never invent measurements, claim
  an unrun check passed, or present an unmeasured tradeoff as a free improvement. If evidence
  does not support an optimization, remove it or keep it explicitly experimental.

### Review before delivery

Read the complete diff against the PR base, including tests, prompts, fixtures and generated
files. Remove experimental residue and verify every changed file serves the stated problem.
Use the PR template to give the mechanism, relevant checks and their actual results, limits
of the evidence, and any correctness, quality, latency or resource tradeoffs.

Do not mark work ready while a neutrality violation is known or a material claim lacks
evidence. Repair the implementation or narrow the claim; an unexplained green score is not
a substitute. Reviewers should cite the specific behavior, code path or missing evidence.
These rules guide implementation and review; documentation and automated review alone
cannot guarantee that every violation is detected.

## What is here

| Path              | What it is                                                              |
| ----------------- | ----------------------------------------------------------------------- |
| `acceleration/`   | Go model router: STT, TTS, LLM and search behind one API, plus the agent that joins a call. `cmd/router` serves it |
| `dashboard/`      | Next.js app for watching calls and editing agent configs. Talks to the router from the browser |
| `sdks/python/`    | The Python SDK: `Agent`, `Runner`, the plugin contracts                  |
| `sdks/js/`        | The JavaScript SDK, one package for Node and the browser: every endpoint typed from the spec, sessions, tools, agent dispatch |
| `plugins/`        | 44 Python packages, one per provider. `plugins/stream` is the client for `acceleration/` |
| `sdks/swift/`     | One iOS package, `core`: state and API, the chat channel, and voice over Stream Video. Its views are Stream Chat's AI components (`StreamChatAI`) |
| `examples/voice_agents/` | Runnable agents. `simple_voice_ai` is the smallest one                  |
| `tui/`            | Go terminal UI for a conversation with an agent: the terminal counterpart of `dashboard/` |
| `sdks/kotlin/` | Client-side SDK for Android: one module, `core`; its views are Stream Chat's AI components |
| `sdks/dart/` | Client-side SDK for Flutter: a core, ui and rtc package |
| `sdks/dotnet/`, `sdks/ruby/`, `sdks/rust/`, `sdks/php/` | Server-side SDKs: agents, dispatch, folder sync and the router. Built and tested in Docker; each has an `sdk-<lang>` skill |
| `sdks/go/`, `benchmark/` | The Go SDK and the voice benchmark. `go.work` ties the Go modules together |
| `performance_benchmark/` | Load benchmark: ramps real WebRTC callers against the router's mock models until it breaks |

## SDK changes

When you need to make an SDK change start by only changing Go
Leave a note in the sdk skill at the bottom that other SDKs also need to udpate.
We run this periodically, you don't need to change all SDKs at once.
Go goes first, and later on we update the others.

## Local dev

Everything reads the repo-root `.env` for provider credentials.

Run the router in Docker, with the Volt dashboard's override (see the `dashboard` skill):

```bash
docker compose -f compose.yaml -f ../volt-dashboard/docs/local-agents/compose.volt.yaml up -d --build router
```

That serves the router on `:8080`, with Postgres on `:55432` and Redis on `:56379`, its data
in the `vision-agents_pgdata` volume. Rerun it after router changes so you never test an old
build. Those two ports are also what the standalone `va-pg` and `va-redis` containers use, so
stop those first if they are running.

`.env` must set `ROUTER_AUTH_KEK` (single-quoted: compose expands a `$` in it) and
`ROUTER_PUBLIC_URL=http://localhost:8080`. Without the KEK no plugin client secret can be
saved, and changing it leaves the stored ones unreadable.

Logs: `docker compose logs -f router`.

An agent, once the router is up:

```bash
cd examples/voice_agents/simple_voice_ai
uv sync && uv run simple_voice_ai.py run
```

It prints a call URL on the dashboard; open it and talk.

The client generators (`plugins/stream/generate.py`, `sdks/swift/generate.py`) are standalone uv scripts with inline metadata: `uv run <script>` does not resolve the workspace.

Python commands all use `uv`. Never `python -m`. If you hit dependency issues, stop and ask.

```bash
uv run --no-sync dev.py check              # ruff + mypy + unit tests
uv run --no-sync pytest -m "not integration"
uv run --no-sync pytest -m "integration"   # needs .env secrets
uv run --no-sync ruff check .
uv run --no-sync ruff format .
uv run --no-sync mypy
```

`--no-sync` avoids a uv panic in sandboxed environments.

## HTTP API

The router serves its API with [chi](https://github.com/go-chi/chi) and [Huma](https://huma.rocks).
The Go structs are the source of truth: `acceleration/api/openapi.yaml` is rendered from them and
is never edited by hand.

- Declare an operation with `huma.Register` in the file for its resource, with its request and
  response bodies as Go structs beside it. `internal/api/policies.go` is the example to copy.
- Describe fields with `doc:` tags and constrain them with `minimum:`, `enum:`, `readOnly:` and
  the rest, so what validates a request is also what documents it. A type's own description goes
  in a `TransformSchema` method, and a named string enum in a `Schema` method using `namedEnum`.
- Fail with an `APIError` (`internal/api/apierror.go`): `invalidRequest(...)`, `notFound(...)`
  and their siblings, whose type decides the status. A failure answered from several places is
  an `APIError` value of its own, named `errX` (`errUnknownConfig`, `errNoSessions`, built with
  `notConfigured(...)` for a feature the deployment lacks), returned as is: one code, one status,
  everywhere. Every failure is the `{"error": {"message", "type", "code",
  "doc_url"}}` envelope; any other error an operation returns is a 500 saying only "something went
  wrong", logged with its stack. A request that fails validation is a 400 `validation_failed`.
- An operation is server-side only unless it sets `Extensions: {"x-client-accessible": true}`.
- List endpoints page by cursor. Read the `pagination` skill
  (`.claude/skills/pagination/SKILL.md`) before adding one or a `limit` parameter.
- After changing an operation, run `go run ./cmd/openapi` in `acceleration/`, then regenerate the
  clients (see [acceleration/README.md](acceleration/README.md)). A test fails if the committed
  `openapi.yaml` is out of date.

- A query parameter a request may leave out is an `optionalParam[T]`; `ptr()` is nil when it was
  left out. Huma takes no pointer for one.
- A route served by hand (a socket, a stream) is still declared, in `internal/api/handwritten.go`,
  so readers and client generators see it and the server-side check reads its marks.

There is no hand-written spec any more: every operation, socket included, is declared in Go.

The JavaScript SDK is its own npm package, checked with node 22 and no runtime dependencies:

```bash
cd sdks/js
npm install
npm run types   # regenerate src/generated/api.ts from the spec; --check in CI
npm test        # typecheck, then the suite against a real http and ws server
```

## Benchmark

To run Voicebench (`benchmark/`) locally or read why a run failed, read the `voicebench` skill
(`.claude/skills/voicebench/SKILL.md`) first: the keys and services a run needs, which command
answers which question, and how to read the report.

To find how many cascaded voice agents one router holds at once, or to measure a performance
change, read the `performance_benchmark` skill (`.claude/skills/performance_benchmark/SKILL.md`):
the ramp in `performance_benchmark/` talks to real agents over WebRTC with the router's `mock/*`
providers, so it bills nothing and measures only the router.

To make the router hold more agents for less, read the `loop_performance_bench` skill
(`.claude/skills/loop_performance_bench/SKILL.md`): ten rounds of profiling, cutting API, SQL and
Redis round trips, deploying to the benchmark cluster and recording agents held and $/voice-minute.

## Connectors

To give an agent a provider's tools (GitHub, Linear, the Slack bot), read the `connectors`
skill (`.claude/skills/connectors/SKILL.md`). To add a provider to the catalog, read
`add-connector`. Plugins are deprecated.

To set up, test or debug an SMS line on a Telnyx number, staging's permanent line included,
read the `telnyx` skill (`.claude/skills/telnyx/SKILL.md`).

## Scheduled tasks

To give an agent reminders and later work for its users, schedule one, or find why one never
arrived, read the `scheduled-tasks` skill (`.claude/skills/scheduled-tasks/SKILL.md`).

## Security

For a security review of the router's API or of the clusters it runs on, read the `security`
skill (`.claude/skills/security/SKILL.md`).

## Testing

For Go tests, read the `go-testing` skill (`.claude/skills/go-testing/SKILL.md`) first: testify
suites, the shared `RouterSuite` for integration tests, and how to wait for async writes.
Before you write Go, read the `go` skill (`.claude/skills/go/SKILL.md`): errors, logging, spans and comments.

Before you run more than one coding agent at once (parallel PRs, reviewers, fixers), read the
`parallel-agents` skill (`.claude/skills/parallel-agents/SKILL.md`): test databases, migration
slots, merge order and how not to burn tokens.

- Framework: pytest. Never mock.
- `@pytest.mark.asyncio` is not needed (asyncio_mode = auto).
- Integration tests use `@pytest.mark.integration`.
- NEVER adjust `sys.path`.
- Keep unit-tests for the class under the same test class. Do not spread them around different test classes. For example, tests for `Agent` must be inside `TestAgent`, etc.
- ALWAYS test behavior, not calling a path.
- Use pytest.fixture for test setup, not helper methods
- NEVER observe method calls in tests; assert on outputs and state.

## Python rules

- Never use `from __future__ import annotations`.
- Prefer specific exceptions if they are known. If the exception type is not clear, it is ok to use `except Exception as e`.
- Avoid `getattr`, `hasattr`, `delattr`, `setattr`; prefer normal attribute access.
- Docstrings: Google style, keep them short.
- Do not use section comments like `# -- some section --`
- Prefer `logger.exception()` when logging an error with a traceback instead of `logger.error("Error: {exc}")`
- Do not use local imports, import at the top of the module
- Avoid `# type: ignore` comments.
- Avoid using `Any` type.
- When adding code to an existing file, follow the patterns already established in that file (e.g. error handling style, import guards, naming).

## Code style

### Imports:

- ordered as: stdlib, third-party, local package, relative. Use `TYPE_CHECKING` guard for imports only needed by type annotations.
- Never import from private modules (`_foo`) outside of the package's own `__init__.py`. Use the public re-export (e.g. `from vision_agents.testing import TestResponse`, not
  `from vision_agents.testing._run_result import TestResponse`).

### Naming:

- private attributes and methods use a leading underscore (`_sessions`, `_warmup_agent`). Public API is plain snake_case.

### Type annotations:

- use them everywhere. Modern syntax: `X | Y` unions, `dict[str, T]` generics, full `Callable` signatures, `Optional` for nullable params.

### Logging:

module-level `logger = logging.getLogger(__name__)`. Use `debug` for lifecycle, `info` for notable events, `error` for failures without a traceback,
`exception` for errors with traceback.

- In hot paths (audio processing, event handling), guard debug logging behind `if logger.isEnabledFor(logging.DEBUG):` to avoid formatting overhead when debug is disabled.

### Constructor validation:

- raise `ValueError` with a descriptive message for invalid args. Prefer custom domain exceptions over generic ones.

### Async patterns:

- async-first lifecycle methods (`start`/`stop`). Support `__aenter__`/`__aexit__` for context manager usage.
- Use `asyncio.Lock`, `asyncio.Task`, `asyncio.gather` for concurrency.
- Clean up resources in `finally` blocks.

### Method order:

- `__init__`, public lifecycle methods, properties, public feature methods, private helpers, dunder methods.

## Plugins

- In every `plugins/*/pyproject.toml`, the wheel target must be `packages = ["vision_agents"]`. Listing `"."` pulls `tests/`, `README.md`, `example/`, etc. into the published wheel.
- Each plugin must keep `readme = "README.md"` in `[project]` and a `README.md` next to its `pyproject.toml` so PyPI renders a description page.

## Token efficiency

- When making multiple related changes to the same file, combine them into fewer Edit calls with enough surrounding context, rather than one edit per change.
- Run tests with Bash directly. Only use subagents for test runs when you need to do other work in parallel.
- Only use TodoWrite for tasks with 5+ steps. Don't update it after every individual edit.

## Changelog

- Lives in `CHANGELOG.md` at the repo root.
- Organised by version heading (`# v0.4.0`), then sections: **Breaking Changes**, **New Features**, **Bug Fixes**.
- Only include user-facing changes (public API breaks, features, fixes). Skip docs-only and CI-only commits.
- Reference PR numbers inline, e.g. `(#374)`.
- To generate: `git log <last-tag>..HEAD --oneline --no-merges`, then classify each commit.
