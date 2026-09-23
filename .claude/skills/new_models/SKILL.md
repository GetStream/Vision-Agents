---
name: new_models
description: Find LLM, TTS, STT and STS models worth adding to the router by searching X and Artificial Analysis, then diffing against what we already route.
---

# What should we be integrating?

The router only wants the frontier and the models everybody is asking for. A weekly
fine-tune, a research checkpoint with no API, a seventh open-weights 3B — none of those are
worth a provider package. The job here is to come back with a short list, not a survey.

## 1. Ask X what shipped

[`cmd/xsearch`](../../../acceleration/cmd/xsearch/main.go) puts a question to Grok with the
`x_search` tool behind it, scoped to a set of handles and a date window. It takes
`XAI_API_KEY` from the environment, or from the nearest `.env` above the working directory,
so it runs the same locally and in a cloud agent.

```bash
cd acceleration
go run ./cmd/xsearch -days 45 -handles OpenAI,OpenAIDevs,AnthropicAI,GoogleDeepMind,xai \
  -query "Which new AI models were launched? For each give the model name, whether it is an LLM, TTS, STT or speech-to-speech model, and the launch date."
```

A call takes a minute or two. Run the batches in parallel, one per shell:

- Labs: `OpenAI, OpenAIDevs, AnthropicAI, GoogleDeepMind, GoogleAI, xai, AIatMeta, nvidia, NVIDIAAIDev, MSFTResearch, Microsoft`
- Voice: `elevenlabsio, DeepgramAI, inworld_ai, cartesia_ai, AssemblyAI`
- Scoreboard: `ArtificialAnlys` on its own, which is where a launch gets ranked rather than
  announced

Write the query so it stands alone. The tool passes the handles separately, so a question
saying "these accounts" makes Grok complain there are none. Widen `-days` when a batch
comes back thin, and drop a handle that returns nothing rather than guessing at its
spelling — accounts get renamed.

## 2. Check it is actually good

X is where a launch is claimed; [Artificial Analysis](https://artificialanalysis.ai) is
where it is measured. Fetch the board for the modality and see whether the model is near
the top or missing entirely:

- <https://artificialanalysis.ai/leaderboards/models>
- <https://artificialanalysis.ai/text-to-speech>
- <https://artificialanalysis.ai/speech-to-text>
- <https://artificialanalysis.ai/speech-to-speech>

Then open the vendor's own docs or changelog for the API model id and the launch date. A
model with a demo and no endpoint is not integrable; say so and drop it.

## 3. Subtract what we have

Everything we route is in
[`router.yaml`](../../../acceleration/internal/routing/router.yaml), one `model:` line per
entry, grouped `stt`, `tts`, `sts`, `llm`. Grep the id before reporting anything:

```bash
rg -n "^      model:" acceleration/internal/routing/router.yaml
```

A model already listed is done. A model that is plainly the next version of one we list is
still worth reporting, as an upgrade rather than a new integration — say which entry it
replaces.

## 4. Report

One block per model, nothing else:

- **Model** — the name, and the API id if it differs
- **Type** — LLM, TTS, STT or STS
- **Launched** — the date
- **Docs** — the page that documents the API
- **Why** — a sentence: where it ranks, what it costs, how fast it is, or who is asking

Close with where each one would go: [router-llm](../router-llm/SKILL.md),
[tts](../tts/SKILL.md), [stt](../stt/SKILL.md), [router-sts](../router-sts/SKILL.md).
