# Voicebench research: STT, TTS, and a stable acceleration bench

This document is the AI-770 research. It maps three benchmark pillars — STT, TTS, and full-agent evaluation — onto what Voicebench and the acceleration provider suites already do, then specifies what to add so the suite can track acceleration while it is being built and, at milestones, stand it next to LiveKit and Pipecat.

The methodology for live agent evaluation lives in [README.md](README.md). This file does not replace it. It says where that methodology is enough, where it is not, and what to build next. The first slice is in tree: `--target accelerated`, `voicebench compare`, `--frozen`, caller-leg WER, turn counts, `voicebench stt`, and `voicebench tts`.

Voicebench scores are comparable to other Voicebench runs under the same setup. They are not comparable to EVA, τ²-bench, eot-bench, Inworld TTS eval, or Pipecat STT numbers. A WER is meaningless unless the dataset, ASR, and normalizer versions all match. The same rule applies to every new pillar.

SIP and telephony are out of scope. Every target joins over WebRTC.

## Two uses

The primary use is a stable instrument that tracks acceleration's progress while it is being built. The question is whether this week is better than last week. That use needs low variance, a frozen definition, and a run cheap enough to do often.

The secondary use is competitive positioning, run at milestones. The question is whether the bundle we ship answers a caller faster and more correctly than what a LiveKit or Pipecat developer actually builds today, which is OpenAI Realtime. That use needs disclosure of each stack's pipeline, not forced sameness of providers.

A metric whose run-to-run spread exceeds the improvement being chased cannot serve either use. Characterizing that spread is the first piece of work, before any new metric.

## What already exists

Voicebench already grades live calls on restaurant, healthcare, and telecom scenarios. A scripted ElevenLabs caller plays over Stream or LiveKit WebRTC. Scoring is deterministic world-state plus a calibrated LLM judge. Reliability is `pass@k` / `pass^k` with a Wilson 95% CI. Timing comes from speech energy in 16 kHz recordings, not from provider-reported clocks. The README already cites EVA, τ²-bench, Full-Duplex-Bench, eot-bench, aiewf-eval, and the open voice-agent benchmark discussion.

The acceleration router already has per-provider integration suites, not public benchmarks:

- [`sttsuite`](../acceleration/internal/stt/sttsuite/sttsuite.go) streams a fixture at call pace and inherits word accuracy (≥90%), settle time, interim arrival, and tail-on-close.
- [`testaudio.Measure`](../acceleration/internal/testaudio/latency.go) reports `ToFirstWords`, `ToSettle`, and `WhileSpeaking`. `ToSettle` is the same quantity Pipecat publishes as TTFS: last speech to the settled transcript.
- [`testaudio.Accuracy`](../acceleration/internal/testaudio/accuracy.go) is `1 − WER` at word level, with case and punctuation folded.
- [`ttssuite`](../acceleration/internal/tts/ttssuite/ttssuite.go) checks TTFB, streaming chunk arrival, audio duration bounds, and barge-in.
- The router records `Turn{STTLatencyMs, LLMTTFTMs, TTSTTFBMs, RoundtripMs}` ([`acceleration/internal/agent/events.go`](../acceleration/internal/agent/events.go)). Python has the equivalent in `observability/collector.py`.

Those suites are pass/fail gates on a single fixture. They do not produce a dataset-level WER, a TTFB percentile table, or a leaderboard.

The registered Voicebench targets in [`internal/target/target.go`](internal/target/target.go) are `python`, `acceleration`, and `livekit`. Neither of the first two matches how agents are written today. See [Comparing acceleration to competitors](#comparing-acceleration-to-competitors).

## Gap analysis

AI-770 asks for six metrics. Coverage today:

| Metric | Status | Where |
| --- | --- | --- |
| Latency | Covered | Recording-based V2V and non-tool reply gap in [`internal/score/timing.go`](internal/score/timing.go). P50/P95 pooled across turns, not a median of per-call medians. Human-band 300–700 ms is reported, not gated. |
| Time to first token | Partial | Router and Python already record LLM TTFT and TTS TTFB on their own turns. Voicebench does not ingest those numbers. Observed V2V is the only cross-target latency. |
| WER | Partial | `testaudio.Accuracy` exists for STT provider tests. Voicebench transcribes with Deepgram Nova-3 for entity/filler/judge input, not for WER. No normalizer, no error-type split, no dataset. |
| Tool calling | Partial | Expected tools, arguments, order, filler during delay, tool wait stats. No wrong-tool-before-right-tool, per-argument precision/recall, hallucinated names, retries, or parallel calls. |
| Goal completion | Covered | `end_state` assertions plus golden-scenario `pass^k`. Binary. No partial credit. |
| Number of turns | Missing | Turn events land in `events.json`. Nothing aggregates turns-to-completion or agent turn length. |

What the six-metric list does not name, and which still blocks the two uses:

| Gap | Why it matters |
| --- | --- |
| No noise-floor measurement | A 40 ms P50 drop is unreadable until the run-to-run spread is known. |
| No cross-run comparison | [`Scorecard`](internal/report/board.go) compares one run to a fixed Voicebench threshold. The CLI is `synth`, `run`, `report`, `calibrate`. Comparing to LiveKit means diffing two `summary.json` files by hand. |
| Stale target roster | The acceleration column measures the raw router API. The shipped SDK-plus-router path is untested. |
| STT/TTS suites are gates, not benches | No dataset, no aggregate reporting, no direct-versus-through-the-router delta. |
| No transcript normalization | Raw WER penalizes `$50` vs `fifty dollars`. |
| No TTS quality or audio health | TTFB is asserted in `ttssuite`. Naturalness, clipping, silence, and MOS are not. |

## A bench stable enough to track development

This is the section that matters most for a stack under active development. No new metric substitutes for it.

### Quantify the noise floor first

Run one unchanged target over the full pack, several times, on a pinned runner and region. Report the spread of every metric. Publish a minimum detectable effect (MDE) per metric: the smallest change the bench can distinguish from noise at the `k` in use.

Until that number exists, "P50 dropped 40 ms" is a claim the instrument cannot support. The MDE decides `k`, the CI wiring, and what `voicebench compare` is allowed to flag. Everything else in this section follows from it.

### Choose `k` from the measured variance

Default `k = 3` is defensible for binary `pass^k` gates. It is thin for a latency percentile. The report already prints the sample count next to every P50, which would expose this: a pack P50 sitting on a handful of measured turns cannot move a trend line.

Latency and correctness may warrant different `k`. A scenario that is all-or-nothing on `end_state` does not need the same sample size as a V2V percentile pooled over turns. The repeatability study should say so rather than picking one `k` for both.

### What is already deterministic

Most of the variance a live-human benchmark would have is already engineered out:

- Caller audio is ElevenLabs `eleven_flash_v2_5`, cached by SHA of voice, text, and sample rate under `cache/tts/`. The stimulus is byte-identical across runs.
- Scenarios are static YAML. Seeded world state is known.
- The judge is pinned `gpt-4.1-mini-2025-04-14`, gated by `voicebench calibrate` against [`calibration/judge.json`](calibration/judge.json). A maintainer must set `reviewed_by`. Calibration requires at least 90% agreement and no missed critical policy violation.
- Evaluator failures (missing TTS, STT, or judge) are `invalid`. They make the scenario incomplete and never count as agent failures.

What remains is provider-side and network-side: model nondeterminism, provider load at time of day, and the network path.

### Attribute the remaining variance

Separate agent variance from evaluator and environment variance. The manifest already carries `network_profile`. Enforce a fixed runner and region for anything that feeds the trend line. A run with a different profile is a different series, not a new point on the same line.

Do not mix dirty trees into the series. `RunManifest.GitDirty` already records this.

### Freeze the bench definition

A metric is only comparable over time if the scenarios, contracts, and thresholds behind it did not move. `MethodologyVersion` (`voicebench-live-v3`), `scenario_hash`, and `contract_hash` already detect drift. Add a policy on top, borrowing Inworld's rule that published presets are immutable and new behaviour means a new file.

Keep a frozen scenario set for trend tracking. New scenarios land outside it until a version bump, so improving the bench never silently rewrites history. Bumping `MethodologyVersion` starts a new series; it does not patch the old one.

### Two run tiers

24 scenarios × `k` × several targets, including two-minute coherence calls, is too slow for per-change feedback.

| Tier | What it runs | When |
| --- | --- | --- |
| Smoke | One golden scenario per pack, `k = 1`, cached caller audio | Per pull request |
| Full | The frozen set at the `k` the MDE implies | Nightly or weekly |

The smoke tier's only cost beyond the agent itself should be the live call. Caller TTS must hit the SHA cache. `--skip-stt` and `--skip-judge` are not a shortcut here: they produce invalid trials by design.

### Regression detection

Store a baseline `summary.json` plus manifest per target under `benchmark/baselines/`, keyed by the acceleration commit, the way eot-bench commits reproducible artifacts under `output/`. The series is a time series, not a folder of one-off runs.

`voicebench compare` then has a baseline mode: diff a run against its stored baseline and flag only changes that exceed the MDE. That is what makes the command a gate rather than a wall of numbers.

### Triage flakiness

Three classes, already almost named by the outcome field:

| Class | Signal | What the operator does |
| --- | --- | --- |
| Evaluator failure | `outcome: invalid` | Fix the evaluator. Do not charge the agent. Do not update the baseline. |
| Infrastructure flake | Valid trial, but `network_profile` drifted, inbound frames dropped, or clock drift outside the usual band | Re-run on the pinned runner. Do not update the baseline from a flake. |
| Agent regression | Valid trial, change exceeds MDE, same profile, same scenario hash | That is the gate. |

Inbound dropped frames already invalidate a trial. Keep that rule.

## Comparing acceleration to competitors

The secondary use, run at milestones. The goal is a defensible head-to-head between the work-in-progress acceleration stack and LiveKit and Pipecat.

### The target roster is stale

The three registered targets do not cover how agents are written today.

The `python` target is pre-acceleration. [`agents/voicebench_agents/restaurant.py`](agents/voicebench_agents/restaurant.py) builds `Agent(edge=getstream.Edge(), llm=openai.Realtime(), instructions=pack_prompt(...))` with `@llm.register_function`. The router is not in the path.

The `acceleration` target in [`internal/target/acceleration.go`](internal/target/acceleration.go) bypasses the Python SDK. It `POST`s to `/v1/agents/sessions` with an inline prompt plus tool schemas and answers `tool_call` frames over a websocket from Go. Function calling, which `Accelerated`'s docstring says stays in Python, is served by the harness. That measures the raw router API, not the product.

Every current example uses a third style neither target exercises. [`examples/agents/customer_support/customer_support.py`](../examples/agents/customer_support/customer_support.py) is `Agent(config="customer_support", llm=acceleration.Accelerated(stt=..., tts=..., model=..., subagent=...))` after `sync_agent(...)`. [`examples/agents/restaurant_orders/restaurant_orders.py`](../examples/agents/restaurant_orders/restaurant_orders.py) is `Agent(config="restaurant_orders")` behind `StreamDispatch`. `Agent(config=...)` fills in edge, llm, agent_user, and phone from the stored config via `_accelerated_defaults`. Instructions, skills, and knowledge come from the agent directory, not from a prompt string.

The scorecard therefore cannot answer the question the suite exists to answer for acceleration: whether the SDK-plus-router path we ship is faster or slower than the plain SDK, and than LiveKit, because the acceleration column measures a different integration than the one customers write.

### The target to add

Add an `accelerated` target: Python SDK plus `stream.Accelerated`, with world tools registered in Python via `register_function` so function calling runs where the docstring says it runs. Reuse the `Runner` / `AgentLauncher` shim in [`agents/voicebench_agents/serve_webrtc.py`](agents/voicebench_agents/serve_webrtc.py) and the existing `python` process-spawn path. This is a small addition, not a new transport.

`accelerated` is the "ours" column and the subject of the trend line.

### What each target isolates

Keep all five. A gap between two columns is only readable if each column has a job.

| Target | What it is | What a gap against `accelerated` means |
| --- | --- | --- |
| `accelerated` | SDK plus `stream.Accelerated`. The shipped bundle. | — |
| `acceleration` | Router session API, tools answered in Go. | SDK overhead on top of the router. |
| `python` | SDK plus `openai.Realtime()`, no router. | Whether the bundle wins with framework and transport held constant. |
| `livekit` | LiveKit Agents on OpenAI Realtime (`gpt-realtime-2`, voice `marin`). | Competing framework, configured the way its users configure it. |
| `pipecat` | Pipecat on OpenAI Realtime. | Same, second competitor. |

### Contract delivery

Voicebench's guarantee is that every target receives an identical prompt and tool set. Config-driven agents read instructions, skills, and knowledge from a directory synced by `sync_agent`. A second hand-written copy of the prompt will drift from [`agents/contracts/`](agents/contracts/).

Render each pack contract into a generated agent directory at run time. Hash that directory into the manifest. That keeps the frozen-definition rule enforceable: a contract change is a new `contract_hash`, which starts a new series rather than silently moving the old one.

### Acceleration competes with OpenAI Realtime

Acceleration is a bundled product, not a pipeline variant. Co-locating STT, LLM, and TTS behind one API so the hops between them disappear is the proposition. `Accelerated` being a cascade (for example Gemini transcribe, Gemini flash-lite, Inworld TTS-2 Flash) while [`agents-livekit/worker.py`](agents-livekit/worker.py) pins `gpt-realtime-2` with voice `marin` is the comparison we want, not a confound to eliminate.

Insisting on a matched provider triple would force acceleration to compete with its bundling switched off and hide the advantage it is built to win on. The headline row is acceleration as shipped against what a LiveKit or Pipecat developer actually builds today, which is OpenAI Realtime.

The as-shipped pipeline is pinned to [`customer_support.py`](../examples/agents/customer_support/customer_support.py): `gemini/gemini-3.5-transcribe-live`, `gemini/gemini-3.5-flash-lite`, `inworld/inworld-tts-2-flash`, subagent `openai/gpt-5.6-sol`. Voicebench does not load that stored config (its instructions and skills are a different product). Changing the triple is a methodology bump.

### Two comparison tiers

Both reported, clearly labeled.

**Tier one** is the product question: which stack answers a caller fastest and most correctly, bundle against realtime, each configured the way its own users would configure it. This is the headline.

**Tier two** is the diagnostic: the same provider triple across all three frameworks, to separate "our bundle is better" from "our framework overhead is lower". It explains a tier-one result. It is not the headline.

Honesty comes from disclosure, not from forcing sameness. The manifest records each target's full pipeline (STT, LLM, TTS, voice). The report states on its face that the headline row compares products, not models. Where a framework cannot express a configuration, mark the cell not comparable rather than publishing a misleading number.

### `voicebench compare`

[`Scorecard`](internal/report/board.go) compares one run to a fixed threshold. There is no head-to-head artifact.

Add `voicebench compare`, following Inworld's `tts-assess compare`: several labeled run directories, one shared threshold set, a per-metric table with the mean and its interval per run, best and worst highlighted, and a marker where a run's interval does not overlap the best. Pass rates already carry Wilson intervals. Latency percentiles are already pooled per pack. The statistical inputs exist.

The same command serves baseline-regression mode from the previous section. One command, two modes: against a stored baseline (progress), or against other labeled runs (competitors).

## STT benchmark

The provider suites already know how to stream a clip at call pace and score the settled transcript. What they lack is a dataset, a declared normalizer, aggregate reporting, and a direct-versus-through-the-router split.

### Metrics

| Metric | Definition |
| --- | --- |
| Pooled WER | Total word errors / total reference words. The number that does not let a short clip dominate. |
| Mean WER | Unweighted average of per-clip WER. |
| Insertion / deletion / substitution | Split of the Levenshtein alignment. `testaudio.Accuracy` already walks that table; it currently returns only `1 − WER`. |
| Perfect-transcript rate | Share of clips with WER 0 after normalization. |
| Transcript-returned rate | Share of clips that produced a settled transcript at all. |
| TTFS median / P95 / P99 | `ToSettle` from [`testaudio.Measure`](../acceleration/internal/testaudio/latency.go). Tail latency matters more than the median. |
| Time to first words | `ToFirstWords`. Whether anything appeared while the caller was still talking. |
| Interim count | `WhileSpeaking`. Live vs. lumped. |

Report raw WER and normalized WER side by side. Normalization cannot be allowed to hide errors.

### Normalization

Gladia's point stands: WER on raw strings penalizes `$50` vs `fifty dollars`, which is formatting, not recognition. Adopt a versioned, declared English subset of their three-stage pipeline: contractions, numbers, currency, symbols, fillers, casefold. Pin the preset name in the manifest. A new preset is a new series.

Do not take a dependency on `gladia-normalization` in the Go harness. Reimplement the small English subset and keep the steps listed, so a reader can see what was folded.

### Semantic WER

Secondary, LLM-judged, using the existing calibrated-judge pattern in [`internal/score/judge.go`](internal/score/judge.go) and `voicebench calibrate`. Semantic WER asks whether a difference would change what a downstream LLM does. Punctuation, contractions, fillers, and equivalent number formats are not errors; names, negations, and numbers that change intent are.

Pin the judge model. Calibrate it on a labeled set the way the policy judge already is. Never let semantic WER replace lexical WER on the scorecard.

### Datasets

Download on demand, hashed, cached like `cache/tts/`. Do not vendor audio.

| Dataset | Why it is in the conversation | Recommendation |
| --- | --- | --- |
| LibriSpeech test-clean | Standard, read speech. | Sanity check, not the headline. Too unlike short agent turns. |
| `pipecat-ai/stt-benchmark-data` | Public, agent-shaped, with published TTFS numbers. | Use it. Running the same providers on the same set is the external check on our instrumentation. Confirm license before depending on it. |
| Common Voice | Accents. | Optional second set, not the trend line. |
| Internal agent-shaped set | Short turns dense in names, numbers, addresses, the entities Voicebench already gates. | Build from existing scenario text plus a small recorded or synthesized set. This is the acceleration-relevant WER. |

The internal set is the trend-line set. The Pipecat set is the published-number check. Mixing them in one pooled WER is a methodology break.

### Router overhead

Run each provider twice: directly, and through the acceleration router. Report the TTFS delta. That isolates what acceleration adds or removes on top of the provider, which no provider-published number can tell us.

The same split belongs on the TTS pillar.

## TTS benchmark

Borrow Inworld's taxonomy. Most of it is cheap in Go. One piece is not.

### Metrics that belong in the harness

| Group | Metric | Source |
| --- | --- | --- |
| Accuracy | Round-trip WER / CER through a pinned ASR | Synthesize, transcribe, compare to the input text. Inworld's caveat applies: ASR errors inflate the score. Read expected vs. heard, and do not treat this as MOS. |
| Audio health | Clipping, loudness proxy, lead/tail silence, tail clicks | Build on [`internal/audio/vad.go`](internal/audio/vad.go) `FrameEnergy`. |
| Latency | TTFB percentiles, real-time factor | Already on `tts.SynthesisComplete` (`TimeToFirstByteMs`, `SynthesisTimeMs`, `AudioDurationMs`). |
| Health grid | Per-clip warn/fail → pass-rate grade | Maps onto the existing gate/verdict model. Inworld grades good ≥99%, warn ≥95%, fail below. Recalibrate those bands on a known-good baseline rather than copying them. |

Same direct-versus-through-the-router TTFB delta as STT.

### MOS

NISQAv2-style MOS prediction is an ONNX model. Go cannot run it cleanly. Three options, none assumed:

1. Skip MOS in the first TTS harness. Ship WER, audio health, and TTFB. Listen to outliers by hand.
2. A pinned Python side-car that the Go CLI shells out to. Adds a Python extra and a model download to a Go module.
3. A curated human listening subset, small enough to re-rate when the trend line moves.

Automated MOS is a proxy, not a listening test. If it lands, it is labeled as such.

### Text corpus

Inworld's 100-utterance English dialogue stress set is MIT. We synthesize the audio ourselves, so we need the text, not their recordings. Supplement with agent-shaped lines pulled from [`agents/contracts/`](agents/contracts/): names, times, confirmation numbers, allergen strings. The stress set catches messy input; the contract lines catch the words acceleration actually has to say.

## Other agent E2E additions

The remaining gaps on the live-call pillar, once the target roster and the comparison command exist.

### TTFT and component attribution

Pull each target's self-reported turn metrics into the trial artifacts: `STTLatencyMs`, `LLMTTFTMs`, `TTSTTFBMs`. Keep them strictly separate from observed V2V. V2V is what the caller heard. Component times are what the stack claims.

LiveKit and Pipecat will not report comparable internals. The cross-framework column stays recording-based V2V. TTFT is a diagnostic for our own stack. Publishing it as a competitor metric would be a methodology break.

### Caller-leg WER

The scripted caller text is already the canonical ground truth for the judge. Scoring Deepgram's caller-leg transcript against that text, through the pillar-1 normalizer, measures Opus encode, loss, and resampling — transport health, not STT quality. A sudden jump is an infrastructure flake, not an agent regression.

### Deeper tool metrics

On top of expected tools, args, and order:

- Wrong tool before the right one.
- Per-argument precision and recall, not only "the tool was called".
- Hallucinated tool names.
- Retry count.
- Parallel calls.

Report them. Do not fold them into the binary pass gate until they have an MDE and a calibration set of their own.

### Turn counts

`events.json` already has scripted turn timestamps. World tool calls have timestamps. Derive:

- Turns to completion (caller turns until `end_state` would pass, if it does).
- Agent turn count.
- Words per agent turn (from the agent transcript).

A bundle that takes twice as many turns to book the same table is slower in a way V2V P50 will not show.

### Partial credit

τ²-bench-style sub-goal checklists, reported alongside the binary `end_state` gate, never replacing it. Speed cannot hide an incorrect booking. Partial credit is for diagnosing *how* a fail happened, not for letting it pass.

### Pipecat

Specified as a competitor column above. Pipecat also publishes STT TTFS on a public dataset; that is the instrumentation check in the STT pillar, not a reason to import their agent scenarios.

### eot-bench

Do not rebuild it. Its unit is a per-pause decision with a false-cutoff / latency Pareto sweep. That does not fit a per-call model. Voicebench already gates barge-in stop, selectivity hold, and false cutoffs on live calls.

If turn detection becomes a differentiator of its own, run an adapter against the Apache-2.0 `livekit/eot-bench-data` set as separate work, not as a Voicebench scenario pack.

### CI

Wire the two run tiers: per-PR smoke, nightly full, cached caller audio, pinned runner and region. Fail a PR on evaluator invalid, on inbound drops, and on a change that exceeds the MDE against the stored `accelerated` baseline. Do not fail a PR on a competitor column.

## Cross-cutting design

### Artifacts

Extend [`internal/report/report.go`](internal/report/report.go) (`SchemaVersion = 2`) to a v3 with a `kind: agent | stt | tts` discriminator. Do not invent a second output format. Keep per-sample JSONL, `summary.json`, and markdown.

### Manifest

Every pillar records, in addition to the fields `RunManifest` already has:

- Dataset hash (or scenario hash, which is the agent-pillar equivalent).
- ASR model, when WER or round-trip WER ran.
- Normalizer version.
- Judge model, when a judge ran.
- Full pipeline triple for agent targets: STT, LLM, TTS, voice.
- Acceleration git commit, when the target is `accelerated` or `acceleration`.
- The MDE table version used to flag regressions, once it exists.

A missing field makes the run incomparable, the same way a missing judge already makes a trial invalid.

### CLI

```
voicebench synth | run | report | calibrate | compare | stt | tts
```

`compare` is the shared cross-run command. `stt` and `tts` produce the same `summary.json` family with `kind` set, so `compare` works on all three pillars.

## Phasing

Driven by getting a usable progress instrument first.

1. `accelerated` target. It is the "ours" column. Existing acceleration numbers describe a different integration.
2. Repeatability study and MDE. Decides `k`. Makes every later number readable.
3. `voicebench compare`, committed baselines under `benchmark/baselines/`, frozen scenario set. The trend line becomes real.
4. Two CI tiers: per-PR smoke, nightly full.
5. `pipecat` target on OpenAI Realtime. Completes tier-one competitor coverage.
6. STT harness reusing `testaudio`, declared normalizer, Pipecat dataset plus internal set, direct-versus-router TTFS.
7. E2E turn counts and deeper tool metrics.
8. Tier-two matched-provider workers, if a tier-one gap needs explaining.
9. TTS harness: round-trip WER, audio health, TTFB percentiles, router delta.
10. MOS, if the open decision goes that way.

## Open decisions

These need a human before implementation treats them as settled.

- How contracts reach config-driven agents. Generated directory from `agents/contracts/` is the recommendation; a maintained parallel tree is the failure mode to avoid.
- Which runner and region the trend line is pinned to. A laptop number and a CI number are different series.
- MOS: skip, Python side-car, or human subset.
- Dataset licensing for `pipecat-ai/stt-benchmark-data` and Inworld's stress-set text, before either is a dependency.
- Whether turn detection gets its own pillar (eot-bench adapter) or stays a live-call gate.
- Whether results are published publicly. Internal baselines can land without that answer. A public leaderboard cannot.

## References

- [Inworld open-tts-eval](https://github.com/inworld-ai/open-tts-eval) — TTS metric taxonomy, immutable presets, offline compare reports.
- [Gladia normalization](https://github.com/gladiaio/normalization) — why raw WER is not a recognition score.
- [Pipecat STT benchmark](https://github.com/pipecat-ai/stt-benchmark) — TTFS, semantic WER, public dataset, Pareto of latency vs. accuracy.
- [ServiceNow EVA](https://github.com/ServiceNow/eva) — `pass@k` / `pass^k`, already in Voicebench.
- [τ²-bench](https://github.com/sierra-research/tau2-bench) — scripted users, stateful worlds, outcome-based success. Partial-credit idea.
- [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) — interruption and overlap. Voicebench covers the live-call slice; do not import the static v1 pipeline.
- [LiveKit eot-bench](https://github.com/livekit/eot-bench) — end-of-turn Pareto. Separate work if we need it.
- [aiewf-eval](https://github.com/kwindla/aiewf-eval) — recording-based voice-to-voice latency, already Voicebench's timing method.
- [Open voice-agent benchmark discussion](https://www.reddit.com/r/voiceagents/comments/1vjdwxj/independent_open_source_benchmark_of_voice_agent/)
