---
name: decision_model
description: What a decision model is in this repo, the contract one implements, the System One protocol every vendor here speaks, and how to add a vendor or a model to the decision_model router. Read before adding a decision model, a decision model vendor or a guardrail backend.
---

# Decision models

A decision model answers named questions about a piece of state with typed values and the
probabilities behind them. It writes no prose. What used to be called an LCM (large classifier
model) here is this; the industry settled on "decision model", and so did we.

The per-modality half of [router-interface](../router-interface/SKILL.md):

| What | Where |
| --- | --- |
| The contract | [`internal/decisionmodel`](../../../acceleration/internal/decisionmodel/decisionmodel.go) |
| The wire protocol | [`internal/decisionmodel/systemone`](../../../acceleration/internal/decisionmodel/systemone/systemone.go) |
| One package per vendor | `internal/decisionmodel/<vendor>/` |
| The router | [`internal/decisionrouter`](../../../acceleration/internal/decisionrouter/router.go) |
| The models | the `decision_model` section of [`router.yaml`](../../../acceleration/internal/routing/router.yaml) |
| The modality | `routing.DecisionModel`, `"decision_model"` in the API and in stats |
| Callers | `/v1/classify`, `type: decision_model` guardrails, prompt injection screening |

## Our definition

Something is a decision model here when all four hold. If one does not, it belongs somewhere
else.

1. **The caller writes the criteria.** It is asked questions the caller wrote, about a state the
   caller chose. A moderation endpoint with a fixed taxonomy (OpenAI moderations, Stream
   Moderation) answers its own categories rather than the question asked, so it is not one.
2. **The answer is a distribution.** `violates: 0.93` is something a threshold acts on. A chat
   model's "Yes, this violates the policy" is something a parser acts on, and a parser that fails
   has no number to fall back to.
3. **The number is meant to be calibrated.** A decision model's 0.9 is meant to be 90%, which is
   what makes a threshold in config mean the same thing across policies. A prompted chat
   model's "0.9" is a token it learned to emit.
4. **It is one request and one answer.** There is no stream, no generated text, no token budget,
   no tools and no history. That is why it routes like `search`, not like `llm`, and why
   failover happens at start only. The caller is mid-turn, and a second provider's latency on
   top of the first one's failure is a longer silence than saying the judgement did not arrive.

A chat model with a JSON schema or logprobs passes the first test and fails the third. It
stays in `llmrouter`. The two run side by side: `internal/guardrail` has a decision model
backend and an LLM-judge backend, and the policy file picks one.

## The three questions

| Constructor | Asks | Answer lands in |
| --- | --- | --- |
| `Noul(instructions, yes, no)` | yes or no | `Answer.Yes`, a probability from 0 to 1 |
| `Choice(instructions, options)` | which of these | `Answer.Chosen`, plus `Probabilities` and `Confidence` |
| `Score(instructions, levels)` | where on this ordered scale | `Answer.Level`, which can fall between two levels, plus `Legend`, `Probabilities` and `Confidence` |

These three are what every vendor below takes. Do not add a fourth unless every vendor's API
has it: a question type that only one provider supports can't be routed. A hazard battery
is several `Noul`s in one request, not a new type. OpenAI's own API calls the noul a
`predicate` and Vercel's AI SDK calls it `boolean`; translate that at the edge, never in the
contract.

Getting the most out of them:

- **A model can only answer with an option it was given.** Include a "none of these" option
  whenever the options may not cover an input. `Choice` keeps an option whose description is
  empty (it goes on the wire as `null`), because the name can be all an option needs.
- **Ask everything at once.** The questions in one request are answered independently and
  share the cost of the state's tokens, so a question whose answer turns out to be irrelevant
  is close to free. Ask for everything you might need and let Go decide what applies. That
  keeps policy in code rather than in a prompt.
- **Use `Score` for a spectrum.** A noul at 0.5 means an even split between yes and no, not
  "medium".
- **`Confidence` says how peaked a distribution is, not whether acting on it is safe.** A
  guardrail should threshold on the probability it cares about, not on confidence.

## The contract

`decisionmodel.Provider` is `Classify`, `Start`, `Close`, `Provider` and `Model`. The rules:

- **Validate first.** Call `Request.Validate()` before anything reaches the wire. A question with
  no instructions, or a type nobody knows, is refused locally rather than spending a round trip
  to come back as a 422.
- **An unanswered question is an error, never a zero value.** A missing `violates` read as `0.0`
  is a guardrail that allows everything, and nothing about the turn would say so.
- **`Result.Model` is the version that answered, not the alias that was asked.** Aliases move
  when a release ships, and a threshold tuned against one version needs to know.
- **Report usage and let config price it.** Report input and output tokens. The configured rates
  turn them into cost; never compute cost in the provider.
- **Map failures onto the contract's errors.** A rate limit (429) wraps `ErrRateLimited`. An
  overloaded service (503, 529) or a network failure wraps `ErrUnavailable`. Anything else is
  the request's own fault. `/v1/classify` turns those into its status codes.
- **No options, ever.** `options.Classifier.Terms()` returns nothing, and that is not a gap to
  close later. Everything a caller asks for travels in the questions, so there is no parameter
  for a provider to accept and quietly drop. A provider that cannot express a question should
  fail the request rather than answer a different one. The caller cannot tell from a
  probability that it was answered about something else.

## The protocol: System One

TypeSafe published System One for Jev. OpenRouter, Perplexity, Alibaba Model Studio and
AnyRouter adopted it unchanged:

```json
{"model": "...", "state": "text, or a JSON object whose parts a question names in backticks",
 "questions": {
   "is_bug":  {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
   "team":    {"type": "choice", "instructions": "...", "criteria": {"payments": "...", "none": null}},
   "urgency": {"type": "score",  "instructions": "...", "criteria": ["lowest level", "...", "highest"]}}}
```

The answers come back keyed the same way, as `{"type", "noul" | "choice" | "score", "probabilities",
"confidence", "legend"}`, beside `model` and `usage.input_tokens` and `usage.output_tokens`.
`systemone.Client` speaks this and nothing else. Perplexity refuses any field it does not
know, so the request carries `model`, `state` and `questions` and nothing more.

OpenAI's own Decisions API (`POST /v1/decisions`, limited preview) is the one shape that
differs: `input` instead of `state`, questions and answers as arrays carrying `name`,
`predicate` with `probability`, and `choices` and `levels` as objects. Reach GPT-6 Luna
Decisions through OpenRouter, which translates it into System One. Write an `openai` vendor
only when there is a reason to skip OpenRouter, and do the translation inside that package.

## The vendors and models

| Vendor | Endpoint | Key | Models declared |
| --- | --- | --- | --- |
| `typesafe` | `api.typesafe.ai/v1/systemone` | `TYPESAFE_API_KEY` | `jev-latest` (the default route's pin) |
| `openrouter` | `openrouter.ai/api/alpha/decisions` | `OPENROUTER_API_KEY` | `typesafe/jev-1.13`, `openai/gpt-6-luna-decisions`, `cloudflare/clef-flash`, `cloudflare/clef`, `respan/span-01-lite`, `liquid/d1`, `jaredpalmer/kev-4b`, `perplexity/pplx-decider-v1.1-27b`, `inception/mercury-decide`, `inception/mercury-decide:free` |
| `perplexity` | `api.perplexity.ai/v1/decisions` | `PERPLEXITY_API_KEY` | `pplx-decider-v1.1-27b`, `pplx-decider-v1-27b` |

To see what OpenRouter currently serves, run
`curl "https://openrouter.ai/api/v1/models?output_modalities=decisions"`. A model can be listed
there with no endpoint behind it, which is why Decider V1 goes to Perplexity directly.

How they are routed:

- **`classify-fast`** is what a guardrail or prompt injection screening gets when it names
  nothing. It prefers `typesafe/jev-latest`, because every threshold in use was tuned against
  Jev. Without the pin, health ranking would hand guardrails to whichever of the ten answered
  fastest last.
- **Clef** is the only `high-quality` model, so it is what the two `*-high-accuracy` shortcuts
  resolve to.
- **Free endpoints** (`:free`) are capped at a few requests a minute and a few dozen a day. They
  are declared `realtime: false`, so a live route never lands on one, and they are priced with
  `free: true` rather than no price. `TestDefaultConfigPricesEveryProvider` still catches a
  price that was forgotten.
- **Data policy.** Only TypeSafe declares one. What happens to an OpenRouter request depends on
  where OpenRouter forwards it, so its models declare none, and a customer who asks for a data
  policy is kept away from them.

## Adding a model

On a vendor that already exists, it is a `router.yaml` entry:

1. Add `provider`, `model` (exactly the vendor's model id), a one-sentence `description`,
   `languages`, `realtime`, `tier` and `price.per_million_input_tokens`, read from the vendor's
   price list and dated in a comment.
2. Choose a tier on purpose. `low-latency` is for the live path. `high-quality` is a slower
   model that is more accurate.
3. Run the integration suite below against it. A model that cannot tell an SDK question from a
   pizza question must not be a candidate for a guardrail.

## Adding a vendor

On System One, a vendor is an `Endpoint` and nothing else:

1. Create `internal/decisionmodel/<vendor>/<vendor>.go` with `ProviderName`, an `Endpoint`
   (key and base-URL env vars, base URL, path, default model) and
   `New(systemone.Options) (*systemone.Client, error)`. Copy `perplexity` as the template.
2. Register it in [`registry.go`](../../../acceleration/internal/decisionrouter/registry.go),
   and add a row to `TestEachVendorIsAskedAtItsOwnEndpointWithItsOwnKey` and a key to the
   `keys` map in `integration_test.go`.
3. Declare its models in `router.yaml`. `TestTheDefaultRegistryHasEveryProviderTheConfigDeclares`
   fails until step 2 is done, which is the point.

Off System One, the package implements `decisionmodel.Provider` itself. It translates the
neutral request into the vendor's wire shape and back inside that package, and follows every
rule under "The contract". The contract carries no vendor's JSON tags, and keeping it that
way is what makes the next vendor cheap.

## Testing

- **Wire behaviour** (validation, criteria shape, unanswered questions, status mapping) is tested
  once, in `systemone_test.go`, against an `httptest` server with a made-up endpoint.
- **A vendor** is covered by the registry test above: right path, right key, right model.
- **`integration_test.go` in `decisionrouter`** (`//go:build integration`) asks every model
  declared in `router.yaml` an obvious yes and an obvious no, and skips a vendor whose key is not
  set. Run it with
  `go test -tags integration ./internal/decisionrouter/ -run DecisionModelIntegration`.
  An OpenRouter account with no credits answers 402 on every paid model, while the
  `:free` one still works.
