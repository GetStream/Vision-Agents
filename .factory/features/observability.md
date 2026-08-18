# Observability

[Sprint 4](../sprint4.md), "Observability".

## Asked for

Per call or session, track the response times of speech-to-text, text-to-speech and the LLM;
the full round-trip delay, which is usually transcribe then answer then speak but can be
shorter with a realtime model; the time to first reply or token; and in future the delay from
voice in to voice out.

## What exists

[turns.go](../../acceleration/internal/agent/turns.go) assembles an exchange as it unfolds and
reports it once, into the `turns` table and onto the agent's event stream. `GET /v1/turns/stats`
serves the rollups in `turn_stats_hourly` and `turn_stats_daily`, with per-leg percentiles.

| Column                 | What it measures                                                 |
| ---------------------- | ---------------------------------------------------------------- |
| `stt_latency_ms`       | The provider's decode time for the transcript that settled the turn |
| `llm_ttft_ms`          | Wait for the first token                                          |
| `tts_ttfb_ms`          | Wait for the first audio                                          |
| `roundtrip_ms`         | Finishing a sentence to hearing the answer start                  |
| `speech_end_to_audio_ms` | Voice in to voice out                                           |
| `audio_out_ms`         | How much speech the turn produced                                 |
| `interrupted`          | Whether the caller talked over it                                 |

Voice in to voice out was the sprint's "in the future" item and is in: it is the round trip
plus the time the transcriber spent deciding the turn was over, since that ran first and the
caller waited through it too.

## A turn is not a request

This is the distinction the feature exists for. A `requests` row measures one provider call. A
`turns` row measures what the caller *felt*: the gap between finishing a sentence and hearing
the answer begin, which spans three providers and the agent's own handling. Neither can be
derived from the other, so both are recorded.

Two consequences:

- **Every leg is nullable.** A realtime model that hears and speaks for itself fills only
  `roundtrip_ms`. An interrupted turn never reaches the legs after the interruption. A leg that
  never happened is left out of the percentiles rather than counted as instant, which is what
  `measured()` does when it turns a zero into a `nil`.
- **A turn closes exactly once.** A reply spoken sentence by sentence produces several
  syntheses, so the tracker waits until the model has finished *and* every synthesis it
  promised has completed. An interruption closes the turn early, and whatever was measured
  before it still happened and is still reported.

## Nothing waits on a database

Turns, transcripts and memories all go through a bounded queue and are dropped when the writer
falls behind. Losing a row costs a statistic; blocking costs the participant a silence. The
turn writer holds 256 rows, bounds each write at five seconds so a stuck database cannot wedge
it, and logs a count of what it dropped on close rather than per drop.

## Not done

Nothing outstanding. Sprint 6 takes this further into evaluation — word error rate against a
slower model after the call, summary scoring, and per-region latency benchmarks — which is a
separate body of work.
