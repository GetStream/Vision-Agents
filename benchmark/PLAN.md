# Voicebench plan: closing the gaps, and provider TTFB/audio tracking

Companion to [RESEARCH.md](RESEARCH.md). That document says what to build and why.
This one says in what order, against what already exists in tree, and it adds a
pillar RESEARCH.md only gestures at: a per-provider TTFB and audio-health series
that is comparable across providers and over time.

Effort is coarse: S is under a day, M is a few days, L is a week or more.

## The constraint that shapes the provider work

RESEARCH.md phase 6 says "STT harness reusing `testaudio`". That is not
implementable as written. `testaudio`, `sttsuite`, `ttssuite`, and both provider
registries live under `acceleration/internal/`, and `benchmark/` is a separate Go
module (`go.work` lists both). Go's `internal` rule means `benchmark/` can never
import them. The reverse is equally blocked: the normalizer
(`benchmark/internal/score/wer.go`, `english-basic-v1`) and audio health
(`benchmark/internal/audio/health.go`) are unimportable from `acceleration/`.

Three ways out, and the one we take:

| Option | Cost |
| --- | --- |
| Duplicate normalizer and health into `acceleration/` | Two normalizer versions that silently diverge. Breaks the rule that the normalizer version pins the series. Rejected. |
| Make `benchmark/` depend on `acceleration/` | Drags the private `getstream-go-webrtc` SSH dependency into the benchmark module, so nobody can build the harness without that key. Rejected. |
| **Split producer from consumer** | One new command, no module surgery. **Take this.** |

**Decision.** Measurement happens in `acceleration/`, scoring and reporting happen
in `benchmark/`, and the contract between them is a JSONL manifest on disk.

A new `acceleration/cmd/probe` drives providers and emits raw measurements. It
never scores. `voicebench stt` and `voicebench tts` ingest those measurements and
own every derived number. This falls out of what already exists: `cmdSTT` already
reads a JSONL of `{id, reference, hypothesis}`. It is already the scorer. What is
missing is the producer.

The split also buys the direct-versus-router arm for free. The probe lives inside
the acceleration module, so it can construct a provider directly through
`sttrouter.DefaultRegistry().Build(...)` *and* drive the same provider through the
router's session API, and report both legs in one manifest.

## Workstream C: make the record correct — done

Smallest work, highest embarrassment if skipped.

| # | Task | Status |
| --- | --- | --- |
| C1 | RESEARCH.md claimed Voicebench does not load a stored acceleration config. It does now, per pack, with skills. | Done |
| C2 | RESEARCH.md recommended rendering contracts into a generated agent directory. We did not: the prompt reaches the router from Python through the same path every other target uses, and the synced directory holds skills only, hashed into `contract_hash`. Recorded as closed, differently, and dropped from Open decisions. | Done |
| C3 | `DefaultLiveKitPipeline` flipped from `inference` to `realtime`, so the default run is the headline product comparison again. The matched triple stays available as the diagnostic behind `VOICEBENCH_LIVEKIT_PIPELINE=inference`. README and both default tests updated to cover each arm. | Done |
| C4 | `RunManifest` now carries `scoring_asr` (`score.ScoringASR`, single-sourced with the Deepgram request URL) and `normalizer_version`. Both are omitted under `--skip-stt`, because a run that scored no transcript has no ASR and no normalizer to declare. | Done |
| C5 | `compare` no longer asserts a fixed "not matched models" sentence. `pipelineDisclosure` reads each run's manifest triple, prints them as a table, and says whether a gap is a product difference or framework overhead. | Done |

## Workstream A: make the instrument trustworthy

Nothing in the agent pillar can be claimed until this lands. RESEARCH.md is blunt
about it: "Until that number exists, 'P50 dropped 40 ms' is a claim the instrument
cannot support."

| # | Task | Effort |
| --- | --- | --- |
| A1 | Repeatability study. One unchanged `accelerated` target, frozen set, `k=1`, five to ten repeats, pinned runner and region. This is mostly machine time, not code. | M |
| A2 | `voicebench mde`: ingest N run directories, emit per-metric spread (min, max, stddev, and the half-width of the repeat interval) and a proposed MDE per metric. Reuses `score.Percentile` and the pooling `summarizeRun` already does. | S |
| A3 | Commit the MDE table as `benchmark/mde.json`, versioned, and record its version in the manifest (RESEARCH.md line 342). | S |
| A4 | Choose `k` per metric family from A1. Binary `end_state` gates and pooled latency percentiles do not need the same sample size. | S |
| A5 | `benchmark/baselines/<target>/<commit>/` holding `summary.json` plus `manifest.json`. Teach `compare --baseline` to resolve a stored baseline by target name rather than only an explicit directory. | M |
| A6 | Extend the MDE gate in `compare` past V2V P50. Today only `--mde-v2v-ms` gates, and pass-rate deltas print with no gate at all. | S |
| A7 | Two CI tiers: per-PR smoke (one golden scenario per pack, `k=1`, caller TTS must hit the SHA cache), nightly full frozen set. Fail on evaluator `invalid`, on inbound drops, and on an MDE-exceeding regression against the stored `accelerated` baseline. Never fail on a competitor column. | M |

A1 gates A2 through A4; A5 through A7 can be built while A1 runs.

## Workstream B: provider TTFB and audio across providers

This is the new pillar. Independent of Workstream A, cheap to repeat, and useful
the day it lands.

### Two sources, never mixed

There is already a production TTFB time series, and it is not the bench.

| Source | What it is | Good for | Why it is not a leaderboard |
| --- | --- | --- | --- |
| Production rollups | `requests` → `latency_p50_ms`/`latency_p95_ms` per `(modality, customer, provider, model, bucket)`, exposed at `GET /v1/stats`. `turns` → `stt_latency_p50/p95`, `llm_ttft_p50/p95`, `tts_ttfb_p50/p95`, `roundtrip_p50/p95/p99` at `GET /v1/turns/stats`. | Drift alarms, live routing health, "is this provider degrading right now". | Confounded by text length, voice, customer, and traffic mix. Provider A serving short confirmations will beat provider B serving paragraphs regardless of speed. |
| Controlled probe (**to build**) | Fixed corpus, fixed voice, fixed rep count, one runner. | Ranking providers, tracking a provider over releases, the router-overhead delta. | — |

Both get reported. The probe is the ranking; the rollups are the monitor. A number
from one must never be published in the other's table.

### B1: the probe command (M)

`acceleration/cmd/probe` with `stt` and `tts` modes. Enumerates provider and model
pairs from `internal/routing/router.yaml`, which is already the authoritative list,
so a newly routed model shows up in the bench without a code change. `--direct`,
`--router`, or both. `--reps N`. Writes a JSONL manifest plus, for TTS, the
synthesized WAVs.

Coverage as of today's `router.yaml`:

- **STT, 7 streaming:** `deepgram/flux-general-en`, `deepgram/flux-general-multi`,
  `gemini/gemini-3.5-transcribe-live`, `parakeet/parakeet-tdt-0.6b-v3`,
  `together-parakeet/nvidia/parakeet-tdt-0.6b-v3-realtime`, `grok/grok-stt`,
  `muse/muse-voice-transcribe-1.0`. Plus `deepgram/nova-3` on the batch path.
- **TTS, 8:** `cartesia/sonic-preview`, `inworld/inworld-tts-2-flash`,
  `elevenlabs/eleven_flash_v2_5`, `elevenlabs/eleven_multilingual_v2`,
  `elevenlabs/eleven_v3_conversational`, `fish/s2-pro`, `s2pro/s2-pro`,
  `breeze/breeze-tts-2`.

The probe reuses `testaudio.Measure` for STT and the existing
`tts.Synthesis.Complete()` metrics for TTS, so the bench reports the same
quantities the router already records rather than a second definition of latency.

### B2: STT metrics (S, on top of B1)

Per `(provider, model, arm)`. Timing fields come from
`testaudio.Timing{SpokeFor, ToFirstWords, ToSettle, WhileSpeaking, Text}`;
everything derived is computed in `voicebench stt`.

| Metric | Source |
| --- | --- |
| TTFS p50 / p95 / p99 | `ToSettle`, last word to settled transcript. The number Pipecat publishes. |
| Time to first words p50 / p95 | `ToFirstWords`. Did anything appear while the caller was still talking. |
| Interim count | `WhileSpeaking`. Separates live streaming from lumped delivery. |
| Pooled WER, raw and normalized | `score.ScoreWER` on `Text` against the fixture reference. Both printed side by side. |
| Substitutions / insertions / deletions | Already on `score.Alignment`. |
| Perfect-transcript rate | Normalized WER of 0. |
| Transcript-returned rate | Non-empty `Text`. A provider that returns nothing is not a fast provider. |
| Router delta | `ToSettle(router) − ToSettle(direct)`, per provider. |

### B3: TTS metrics (M, on top of B1)

Per `(provider, model, voice, arm)`. `TimeToFirstByteMs`, `SynthesisTimeMs`, and
`AudioDurationMs` come off `tts.SynthesisComplete`; health comes from
`audio.MeasureHealth` in `voicebench tts`.

| Metric | Source |
| --- | --- |
| TTFB p50 / p95 / p99 | `TimeToFirstByteMs`. |
| Real-time factor p50 / p95 | `SynthesisTimeMs / AudioDurationMs`. Throughput after the first byte. |
| Inter-chunk gap p95, longest stall | Arrival timestamps of `tts.AudioChunk`. **New.** `ttssuite` only asserts more than one chunk arrived, so a provider with a fast first byte that then starves the jitter buffer currently scores clean. This is the metric most likely to explain a bad call that TTFB says is fine. |
| Clipping, RMS, peak | `Health.ClipFraction`, `Health.RMS`, `Health.Peak`. |
| Lead / tail silence, silence ratio | `Health.LeadSilenceMS`, `Health.TailSilenceMS`, `Health.SilenceRatio`. Dead air at the head of a turn reads to the caller as latency and is invisible to TTFB. |
| Round-trip WER | `score.TranscribeDeepgram` then `score.ScoreWER` against the input text. Inworld's caveat is reported on the table: ASR error inflates this, and it is not MOS. |
| Duration error | `AudioDurationMs` against the expected band for the corpus line. Catches truncation and runaway generation. |
| Failure rate, empty-audio rate | Synthesis errors and `AudioDurationMs == 0`. |
| Router delta | `TTFB(router) − TTFB(direct)`, per provider. |

### B4: corpus (S)

STT reuses the `tests/test_assets` fixtures the suites already use, extended with
agent-shaped clips dense in names, numbers, and addresses, since those are the
entities Voicebench already gates. TTS text comes from `agents/contracts/`:
confirmation numbers, times, allergen strings, the words acceleration actually has
to say. Both corpora hashed into the manifest. Public datasets
(`pipecat-ai/stt-benchmark-data`) stay a separate, clearly labeled series for
checking our instrumentation against published numbers, and are blocked on the
licensing question.

### B5: summary emission (S)

`voicebench stt` and `voicebench tts` currently print to stdout and write nothing.
Have them emit `summary.json` with `Kind` set to `KindSTT` / `KindTTS`. Those
constants exist in `report.go` and are never assigned today, which is why
RESEARCH.md line 352's promise that `compare` works on all three pillars is not
true yet. Once they are set, `compare` works on provider runs with no change to it.

### B6: health bands (S)

Grade each clip warn/fail into a pass-rate, mapping onto the existing gate model.
Derive the bands from the first full probe run, per RESEARCH.md line 256, rather
than copying Inworld's 99/95 numbers onto providers we have not characterized.

### B7: repeatability for the probe (S)

Same discipline as A1 but far cheaper, because a rep is one five-second clip rather
than a full call. Rule for publishing: p95 needs at least 30 reps, p99 needs about
100. A p99 off ten reps is noise wearing a percentile's name.

## Workstream D: remaining coverage

Only worth doing once A and B make the numbers readable.

| # | Task | Effort |
| --- | --- | --- |
| D1 | `pipecat` target on OpenAI Realtime. Completes tier-one competitor coverage. | L |
| D2 | Deeper tool metrics: wrong tool before the right one, per-argument precision and recall, hallucinated names, retries, parallel calls. Reported, not folded into the pass gate until each has an MDE. | M |
| D3 | Ingest each target's self-reported `STTLatencyMs`, `LLMTTFTMs`, `TTSTTFBMs` into trial artifacts, kept strictly separate from observed V2V. Diagnostic for our stack only; publishing it as a competitor metric would be a methodology break. | M |
| D4 | Tier-two matched-provider workers, if a tier-one gap needs explaining. Partly already done, see C3. | M |
| D5 | τ²-style partial-credit sub-goal checklists alongside the binary `end_state` gate. | M |
| D6 | MOS, if the open decision goes that way. | L |

## Sequencing

```
done     C1..C5                      correct the record
next     A1 (machine time) ──┐
                             ├─ A2 A3 A4        MDE, k
         B1 ── B2 B4 B5 ─────┘                  provider bench usable
              └ B3 B6 B7                         TTS depth, bands
then     A5 A6 A7                                baselines, gated CI
later    D1..D6
```

Workstream B does not wait on A. Different variance regime, its own cheap
repeatability, and it answers a question we get asked constantly: which provider
is fastest, and what does the router add on top.

## Open decisions

Needs a human before implementation treats any of these as settled.

- ~~**C3, the headline definition.**~~ **Resolved:** the headline is acceleration as
  shipped against OpenAI Realtime, matching RESEARCH.md. Matched providers are the
  tier-two diagnostic, reachable with `VOICEBENCH_LIVEKIT_PIPELINE=inference`.
- Which runner and region the trend line is pinned to. A laptop number and a CI
  number are different series, and every number we have so far is a laptop number
  at `k=1`.
- Whether the probe's `--router` arm goes through a local router or a deployed one.
  Local isolates provider and router; deployed includes network path and is what a
  customer experiences.
- MOS: skip, Python side-car, or human listening subset.
- Dataset licensing for `pipecat-ai/stt-benchmark-data` and Inworld's stress-set
  text, before either becomes a dependency.
- Whether results are published externally. Internal baselines can land without
  this answer; a public leaderboard cannot.
