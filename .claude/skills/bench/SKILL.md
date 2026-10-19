---
name: bench
description: Benchmark the accelerate branch end to end with one command, either a short bench that covers the basics of a voice agent (about 10 minutes) or every benchmark (about 45 minutes), using Voicebench and the STT/TTS benches through the Go SDK. Use when asked to benchmark, validate or smoke-test accelerate, run all the benchmarks, run a short or quick bench, or check a change before merging.
---

# Bench the accelerate branch

`benchmark/scripts/bench.sh` runs the benchmark infrastructure against a checkout of
`accelerate`, one stage after another, and writes `SUMMARY.md` with every stage's status and
where its report is. It wraps what already exists; the details of each stage live in the
`voicebench` skill, read it when a stage fails.

| Mode | Stages | Calls | Time | Cost |
|---|---|---|---|---|
| `short` | `agents-basics` | 12 | ~10 min | Under a dollar |
| `full` | `components`, `agents-frozen`, `agents-extended` | 36, plus every caller and agent line through STT and TTS | ~45 min | A few dollars of ElevenLabs, Deepgram, OpenAI, Gemma |

## The short bench: the basics

`scenarios/basics.txt` is one call for each basic behavior, four a pack so the three packs,
running side by side, finish together:

| Behavior | Scenario |
|---|---|
| Completes a booking end to end | `restaurant.golden` |
| Stops when the caller cuts in and takes the change | `restaurant.interrupt` |
| A child talks to the parent, the parent answers the child, then comes back: the agent answers neither | `restaurant.side_voice` |
| The caller corrects a detail mid-sentence | `restaurant.correction` |
| A television talking in the room | `healthcare.tv` |
| Nine seconds of silence: the agent checks in | `healthcare.silence` |
| A minute-long turn with thinking pauses: no cutting in | `healthcare.monologue` |
| A request the policy forbids | `healthcare.adversarial` |
| Filler while a slow tool runs | `telecom.tool_filler` |
| Street noise | `telecom.noise` |
| Coughs and side talk ignored, a real interruption accepted | `telecom.selectivity` |
| Numbers read digit by digit, a spelled street | `telecom.dictation` |

To cover another basic, add its scenario to `basics.txt` and keep four a pack (the test
`TestBasicsListIsFourScenariosAPack` holds it). A new scenario goes into `extended.txt` too;
see the `voicebench` skill, section 6.

## The full bench

- **`components`**: STT on every scenario's caller lines and TTS on their agent lines, streamed
  through the router's sockets by the Go SDK (`sdks/go`, `scripts/components.sh`).
- **`agents-frozen`**: the trend-line set, plus the agent's STT alone on its caller lines.
- **`agents-extended`**: the harder callers and rooms outside the frozen set.

Every Voicebench stage calls our stack, the Python agent on `stream.Accelerated` over a router
built from the checkout, the three packs side by side (`scripts/digest.sh`).

## 1. Before running

1. **Be on `accelerate`**, pulled: `git -C <repo> switch accelerate && git pull`. The script warns
   when it is not, and every report records the branch, commit and how many tracked files were
   modified: a dirty checkout benchmarks the local changes, say so in the report.
2. **Say what it will cost** and, for `full`, ask before starting. `short` needs no asking. A
   single stage: `BENCH_STAGES="agents-frozen" scripts/bench.sh full`.
3. **Native libraries**: `brew install pkg-config opus opusfile libsoxr`; Docker running.

The script checks the rest itself and stops before any call if a check fails: the keys in the
`.env` voicebench reads (`benchmark/.env`, else the repo root; names only, never print a value),
the ports the stages use (8080, 8001-8009, 8090-8099), Postgres and Redis through
`docker compose`, and Gemma awake (it waits up to ten minutes for a cold deployment). If 8080 is
the compose `router` service, stop it for the run (`docker compose stop router`) and start it
again after; ask before stopping anything else. A key that is present but dead is not caught
there: on a run where every call comes back invalid, run the key probes in the `voicebench`
skill, section 1.

## 2. Run

From `benchmark/`, in the background, following its output:

```bash
scripts/bench.sh short
scripts/bench.sh full
```

Results go to `out/bench-<mode>-<time>/` (a second argument names another directory). Nothing
is posted to Slack and LiveKit is not run: set `VOICEBENCH_LIVEKIT_ARMS=inference` to add the
LiveKit Inference arm, `VOICEBENCH_STS_ARMS=openai/gpt-live-1` to add GPT-Live.
`VOICEBENCH_K=3` repeats every scenario (triples the time).

**Watch it.** Each call prints `voicebench: [3/4] restaurant.side_voice: started` and then its
verdict. No new line for six minutes is a stuck call: `digest.sh` kills a pack at its timeout by
itself, but if the whole script hangs, kill its tree (`kill_tree` in the `voicebench` skill) and
check the ports again before rerunning.

## 3. Read the results

Start from `SUMMARY.md`, then each stage's report:

| Stage | Read | Look for |
|---|---|---|
| `agents-*` | `<stage>/voicebench.html`, `<stage>/<pack>-accelerated/` | Passed out of valid per pack, failure causes, reply-time P50/P95 with sample counts |
| `components` | `components/stt/report.md`, `components/tts/report.md` | WER and TTFS per STT target; TTFB, WER and health grid per TTS target |

`voicebench.html` sorts failures by cause: **infra** is setup, never the agent; then heard
wrong, did wrong, said wrong, turn-taking. Match failures against the known signatures in the
`voicebench` skill before investigating, and use its section 4 to look at one call. For the
short bench, name the basic each failed call stands for.

A stage that **failed** in `SUMMARY.md` means its script exited non-zero: a pack that timed out,
an invalid trial, a router that did not start, an STT or TTS clip that errored. Its logs are in
the stage directory. Calls that ran and failed their gates are in the report.

## 4. Did a change help?

The short bench says whether a basic broke; at `k=1` it is not evidence that something got
better. For that, run the same stage on the base commit and the change back to back with
`VOICEBENCH_K=3`, then `go run ./cmd/voicebench compare --baseline <base>/<pack>-accelerated <new>/<pack>-accelerated`.
Section 5 of the `voicebench` skill says what counts as a difference.

## 5. Report back

```
Bench <short|full> on accelerate@<sha> (<clean|n modified files>)
Voicebench <set>: <passed>/<valid> passed, <n> invalid (<why>); restaurant <p>/<v>, healthcare <p>/<v>, telecom <p>/<v>
Basics broken: <behavior (scenario)>, ... | none
Reply time: P50 <ms> P95 <ms> over <n> non-tool turns
Components: STT <target> WER <x>%, TTFS P50 <ms>; TTS <target> TTFB P50 <ms>, <health>
Failures:
- <scenario>: <cause>. Evidence: "<quoted tool call, heard line, judge note or log line>"
Next: <the one thing to fix or rerun>
```

Leave out the lines for stages that did not run. Keep setup failures out of the scores and say
what they were.
