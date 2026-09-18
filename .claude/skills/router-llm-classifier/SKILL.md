---
name: router-llm-classifier
description: What the classifier router can be asked for, why it is not the LLM router, and what it refuses to fake. Read before adding a classifier provider or a guardrail backend.
---

# Routing classification

The per-modality half of [router-interface](../router-interface/SKILL.md). The contract is
[`internal/llmclassifier`](../../../acceleration/internal/llmclassifier/llmclassifier.go), the
providers sit under it, and they are declared in the `llm_classifier` section of
[`router.yaml`](../../../acceleration/internal/routing/router.yaml).

A classifier answers a question about a piece of text with a typed value and the probability
behind it. There is no stream, no generated text and no token budget: one request, one set of
answers, so it routes like `search` rather than like `llm`. Failover is start-time only for the
same reason it is for a model — a caller is mid-turn, and a second provider's latency on top of
the first one's failure is a longer silence than saying the judgement did not arrive.

## Why this is not a mode of the LLM router

You can get a yes or no out of any chat model with a narrow prompt and a JSON schema, and it
would route through `llmrouter` today. The reason not to:

- **A distribution is the answer, not a sentence about the answer.** `violates: 0.93` is
  something a threshold can act on. "Yes, this appears to violate the policy" is something a
  parser can act on, and a parser that fails has no number to fall back to.
- **Calibration is the product.** A prompted model's "0.9" is a token it learned to emit. A
  classifier's 0.9 is meant to be 90%, which is what makes a threshold in config meaningful
  across policies rather than tuned per prompt.
- **The request shape is different.** A caller asks named questions about a named state, so a
  question can point at `` `message` `` and be answered independently of its neighbours. There
  is no instruction string, no tools, no history, no format.

The two go side by side rather than one inside the other: `internal/guardrail` has both a
classifier backend and an LLM-judge backend, and the policy file picks.

## The primitives

Three, and they are the classification primitives rather than one vendor's vocabulary:

| Constructor | Asks | Answer lands in |
| --- | --- | --- |
| `Noul(instructions, yes, no)` | yes or no | `Answer.Yes`, a probability from 0 to 1 |
| `Choice(instructions, options)` | which of these | `Answer.Chosen`, plus `Probabilities` and `Confidence` |
| `Score(instructions, levels)` | where on this scale | `Answer.Level`, which may fall between two levels |

Two things that are easy to get wrong:

- **An option a classifier was not given cannot be answered with.** Include something for "none
  of these" whenever the options may not cover an input. `Choice` keeps an option whose
  description is empty, because a name can be the whole of what an option means.
- **Ask everything at once.** Questions in one request are answered independently and share the
  state's tokens between them rather than paying for it each, so a question whose answer may
  turn out to be irrelevant is close to free. Ask for what you might need and let Go decide
  what applies — that is what keeps policy in code instead of in a prompt.

`Confidence` says how peaked a distribution is. It does not say whether acting on it is safe,
and a guardrail that thresholds on confidence rather than on the probability it cares about is
measuring the wrong thing.

## The vendor landscape, honestly

[router-interface](../router-interface/SKILL.md) asks for the top five, and there is no
five-vendor market here. What exists:

| | Caller-defined criteria | Calibrated probability | Shape |
| --- | --- | --- | --- |
| TypeSafe System One (Jev) | yes | yes, that is the product | named questions over a state |
| A chat model with a JSON schema | yes | no, a token that looks like one | prompt and parse |
| A chat model's logprobs | yes | roughly, over its own tokens | prompt and read the top logprob |
| OpenAI moderations | no, fixed taxonomy | per-category scores | one text in, fixed categories out |
| Stream Moderation | no, fixed policy | per-category scores | one text in, fixed categories out |

The moderation endpoints are a different product: they answer their taxonomy, not the question
you asked, so they cannot serve a policy written by a customer. Do not add them here to make the
table longer — a provider that ignores the questions and answers its own categories is exactly
what `Terms()` exists to prevent elsewhere.

The second provider worth building is an `openaicompat` classifier that asks a chat model a
single-token question and reads the top logprobs, so the interface is held to shape by something
that is not TypeSafe. It is uncalibrated and should say so in its declaration rather than in a
comment.

## What the router refuses to fake

`options.Classifier.Terms()` returns nothing, and unlike `search` that is not a gap to close
later: everything a caller asks for travels in the questions themselves, so there is no
parameter for a provider to accept and silently drop. A provider that cannot express a
question type should fail the request rather than answer a different question — the caller
cannot tell from a probability that it was answered about something else.

`Request.Validate()` is called by each provider before anything reaches the wire, following the
same pattern as `search.Query.Validate()`. A question with no instructions or a type nobody
knows is refused locally rather than spending a round trip to come back a 422.

An unanswered question is an error, never a zero value. A missing `violates` read as `0.0` is a
guardrail that allows everything, and nothing about the turn would say so.

## Adding a provider

1. A package under `internal/llmclassifier/<vendor>/` implementing `llmclassifier.Provider`:
   `Classify`, `Start`, `Close`, `Provider`, `Model`. Translate the neutral request into the
   vendor's wire shape in that package — the contract carries no vendor's JSON tags, and
   keeping it that way is what makes the second provider cheap.
2. Register it in
   [`registry.go`](../../../acceleration/internal/llmclassifierrouter/registry.go).
3. Declare it in `router.yaml` with `languages`, `tier`, `data_policy` and a price. Bill
   `per_million_input_tokens` unless the vendor bills otherwise; the session reports usage and
   lets the configured rates price it.
4. An httptest wire test, and an `//go:build integration` test that asserts the model answers
   an obvious yes and an obvious no in opposite directions. The registry-covers-the-yaml test in
   `llmclassifierrouter_test.go` will start failing until step 2 is done, which is the point.

## Adding a question type

Don't, unless a vendor's own API has one. The three here are what the calibration research
behind Jev is built on, and a fourth that one provider supports is a question the router cannot
route. A hazard battery is several `Noul`s in one request, not a new type.
