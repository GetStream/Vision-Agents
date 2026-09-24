---
name: refresh_model_stats
description: Refresh the Artificial Analysis numbers (TTS Elo and characters per second, streaming STT word error rate and latency) that the model picker shows. Use when the board has moved, a model was added to router.yaml, or someone asks for fresh model stats.
---

# Refresh model stats

The model picker in Volt shows benchmark columns for voice and transcription models. They
come from `benchmark:` on each model in `acceleration/internal/routing/router.yaml`, which
`GET /v1/{modality}/providers` returns as `benchmark`. Nothing fetches them live; this skill
is how they change.

| Modality | Fields | Artificial Analysis source |
| --- | --- | --- |
| tts | `elo`, `characters_per_second` | [text-to-speech](https://artificialanalysis.ai/text-to-speech), arena Elo and median characters per second |
| stt | `word_error_rate` (0 to 1), `latency_ms` | [speech-to-text/streaming](https://artificialanalysis.ai/speech-to-text/streaming), AA-WER streaming index and time to final transcript |
| search | `search_index` (0 to 100), `cost_per_task` (USD) | [agents/search-api](https://artificialanalysis.ai/agents/search-api), Search Index and search plus model cost per task |

LLMs only show popularity, so they carry no benchmark.

## 1. Pull the numbers

```bash
uv run --no-project .claude/skills/refresh_model_stats/aa_stats.py
```

It reads the data embedded in both pages and prints one tab-separated row per model and
host. If it prints nothing, the page shape changed: fetch the page with `curl`, find the
field names (`qualityElo`, `medianCharactersPerSecond`, `aaWerStreamingIndex`,
`timeToFinalTranscriptSeconds`) and fix the script.

## 2. Match rows to our models

Match on what we actually call, not the closest name:

- A versionless model id takes the version the vendor serves today. `grok/grok-stt` is
  Grok Voice Transcribe 2.0; `cartesia/sonic-preview` is Sonic 3.6 (see its comment).
- Pick the variant our code runs. `cartesia/ink-2` is the socket that detects turns itself,
  so "semantic endpoints"; Nemotron on Together is the "Together AI" row.
- Arena Elo belongs to the weights, speed to the host. A model we self-host (`s2pro`,
  `breeze`) gets `elo` only, since AA measured someone else's servers.
- A search variant is the setting we send. Exa's `fast` and `auto` and Tavily's `basic`
  map by name. `perplexity/search` sends no `search_context_size`, and Perplexity's docs
  disagree on whether that means `low` or `high`, so it stays blank until we pin one.
- A model AA has not measured gets no `benchmark` line. Do not borrow a sibling's numbers
  (`flux-general-multi` is not `Deepgram Flux`).

## 3. Write them

One line after the model's `description`, rounded the way the script prints them:

```yaml
      benchmark: { elo: 1273, characters_per_second: 115 }
      benchmark: { word_error_rate: 0.027, latency_ms: 490 }
      benchmark: { search_index: 74, cost_per_task: 0.127 }
```

Then, in `acceleration/`:

```bash
go test ./internal/api/ ./internal/routing/ -count=1
```

Restart the router so the picker sees the new numbers. Mention in the commit which models
moved and the date the board was read.
