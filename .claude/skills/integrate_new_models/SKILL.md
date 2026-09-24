---
name: integrate_new_models
description: Find the models worth adding with the new_models skill, then build support for each one in its own branch and open a draft PR per model.
---

# From "we should have this" to a PR per model

[new_models](../new_models/SKILL.md) says what is missing. This does the work, one model at a
time, one branch at a time, so each lands as a PR that can be judged on its own.

## 1. Get the list

Run [new_models](../new_models/SKILL.md) and keep its report. It has already dropped anything
in [`router.yaml`](../../../acceleration/internal/routing/router.yaml), so what is left is the
work. Drop anything else the report cannot give an API id and a docs page for — an integration
cannot be written against a blog post.

## 2. Decide how big each one is

Two very different jobs wear the same words. Before dispatching anything, check whether the
vendor already has a package for that modality — `internal/tts/<vendor>`, `internal/stt/<vendor>`,
`internal/llm/<vendor>`, `internal/sts/<vendor>`:

- **The package exists.** A new model from a vendor we already speak is usually a `router.yaml`
  entry plus whatever the new model does that the old one could not. Small.
- **The package does not exist.** A new vendor, or a modality we have never asked that vendor
  for, is a provider package, its registry entry, three test files and the config. Gemini TTS is
  this even though we already talk to Gemini for STT and speech-to-speech.

Say which of the two each model is in the report at the end. It is the difference between an
afternoon and a week, and it is the first thing a reviewer wants to know.

## 3. One subagent per vendor per modality

Dispatch a subagent per unit of work, in parallel, each with the model `claude-opus-5-5-medium`.
Cloud subagents based on the working branch are the way to do it: each takes its own branch and
VM, and the cloud environment carries the vendor API keys as secrets, so the integration tests
can actually run. Locally, `best-of-n-runner` gives the same isolation through a worktree.

The unit is one vendor's models in one modality, not one model. Three OpenAI LLMs are one PR,
because three PRs editing the same lines of the `llm:` block conflict with each other and none
of them can be merged on its own. Two vendors are always two PRs even in the same modality.

Branch per unit: `model/<vendor>-<what>`.

Each subagent gets: the name, API id, type, docs URL and launch date of every model in its unit;
which of the two jobs in step 2 it is; and the skill to follow.

| Type | Skill to follow |
| --- | --- |
| TTS | [tts](../tts/SKILL.md) to build it, [router-tts](../router-tts/SKILL.md) for what config may claim |
| STT | [stt](../stt/SKILL.md) to build it, [router-stt](../router-stt/SKILL.md) for what config may claim |
| LLM | [router-llm](../router-llm/SKILL.md) |
| STS | [router-sts](../router-sts/SKILL.md) |

Say which handed-over facts are unverified. An announcement gives a marketing name and a launch
date; it rarely gives the API model id, and almost never gives a price. Those two are what the
config is made of, so a subagent told "the id is probably X" will write X and move on. Name the
gap instead and tell it to settle the question from the vendor's own docs and report what it
found — `router.yaml` treats price as something this deployment is billed, not a guess.

Tell each subagent to finish with a draft PR per [pr](../pr/SKILL.md), and to commit per
[commit](../commit/SKILL.md). A subagent that cannot get the model working must still say so and
leave the branch unpushed rather than open a PR on a guess — a router entry for a model nobody
has heard answer is worse than no entry, because routing will send real traffic at it. The same
goes for a green-looking PR over a failing live suite.

When a live suite fails on a brand-new API, ask the API rather than reason about it. A throwaway
program that opens one session and prints every frame settles in a minute what a day of reading
the provider will not, because the docs of a just-launched model are thin and the failure is
usually a fact about the protocol nobody wrote down. Unit tests over a fake server cannot find
these: the fake answers the way the author expected, which is the thing in doubt. GPT-Live turned
out to run on a media clock that only input audio advances, so every path that handed it text
hung forever — invisible to a fake, obvious in two frames of the real thing.

## 4. Report

Per model: the name, the type, which of the two jobs it turned out to be, the PR link, and
whether the integration test passed. Then the models that were dropped and why, so the next run
does not re-litigate them.
