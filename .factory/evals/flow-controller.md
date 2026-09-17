# The flow controller benchmark

Which model should decide whether a caller has finished, whether they were talking to the agent
at all, and whether they have taken the floor. The twelve situations it decides between are
[the conversation](../features/conversation.md)'s; this is the labelled set for asking whether
Gemma 4 is the right thing deciding them, and whether TypeSafe's Jev is better.

The reason to ask is that these are the cheapest judgements on the live path and the ones with
the shortest fuse. Every one of them happens while somebody is waiting, and getting one wrong
is not a worse answer but a worse call: an agent that talks over a correction, or one that
stops mid-sentence for a cough.

## What is compared

| Arm | What it is |
| --- | ---------- |
| `gemma` | `cerebras/gemma-4-31b` given the production `flowInstructions` and `flowQuestion`, parsed by the production `parseFlow`. The incumbent, unaltered |
| `jev-direct` | `jev-latest` asked two Choice questions in one request, the closest thing to what Gemma is asked |
| `jev-composed` | The same two Choices plus four Nouls in the same request, with the policy the conversation already hard-codes applied in Go |

Gemma answers three times per case, because it samples and a controller that flips between two
answers for the same words flips mid-call. Jev answers once, because it returns the distribution
rather than a draw from it.

The composed arm exists because the interesting question is not only which model classifies
better. It is whether the judgements the conversation already makes in code — a noise does not
take the floor, a menu is never cut off, words that were not for the agent are not answered —
are better asked for separately and combined, which is what TypeSafe's own guidance says to do.
Its three overrides are those three and no more, and every threshold in it is the neutral half,
because a number chosen against this set is a number this set can no longer measure.

## The set

120 cases, ten per state, hand written from the semantics of the production prompt rather than
mined from calls, split evenly between the six states with the floor free and the six with the
agent talking. Three domains, matching voicebench's packs: a restaurant, a clinic and a
telecoms provider, plus an outbound call into somebody else's menu.

It lives in
[testdata/flowbench.json](../../acceleration/internal/harness/testdata/flowbench.json) and is
checked by `TestFlowSetSuite`, which runs with the unit tests: every state covered ten times,
the two halves balanced, no case labelled something its state does not want, and no noise case
that the agent's own `overlapNoise` would have caught before any model was asked. That last one
matters most. A benchmark whose non-speech cases are the ones a fixed word list already catches
is measuring the word list.

`recover` — a wait the conversation has seen too many times, which it answers with a question
instead — is not in the set. It is decided from a clock and no model is ever asked about it. The
twelfth state is `wait-digits` instead, the half-said PIN or clock time the prompt singles out.

## What is graded

The outcome, not the JSON. A disposition and a floor are not separately gradeable because the
conversation reads them together and one overrides the other: an `ignore` takes the floor
decision away whatever it said. So each arm's two answers are put through the same reading
`converse.Ruled` and `converse.overlapRuled` take, and what is scored is what the agent would
have done — `answer`, `answer-clarify`, `wait`, `ignore`, `interrupt`, `shorten` or `continue`.

Reported per arm:

- Correct overall, and split by whether the floor was free or held, which is the same split as
  which axis was doing the deciding.
- Correct per state, so a model that fails only on menus can be told from one that fails
  everywhere.
- A confusion matrix of what was wanted against what would have happened.
- **Missed stop**: a caller took the floor and did not get it. This is the agent talking over a
  correction, and it is the failure a caller hangs up over.
- **False stop**: the agent gave the floor up to something that was not asking for it.
- **Unreadable**: answers that had to fall back to what the controller does with JSON it cannot
  parse. Gemma's fallback is scored as the outcome it produces, so a truncated answer counts as
  the behaviour it causes rather than as a hole in the table.
- **Flipped**: cases whose repeats disagreed with each other.
- p50 and p95 against the production `flowDeadline` of 3s.
- Cost per thousand decisions, from the router's own prices for Gemma and $42 per billion input
  tokens for Jev, whose output tokens are free.
- For the Jev arms, the share of judgements whose own confidence came back under 0.6, which is
  what a deployment would escalate rather than act on. It is reported and not applied: what to
  do with an uncertain judgement is a decision about the call, not about the model.

## Running it

```bash
cd acceleration
FLOW_BENCHMARK=1 go test -tags integration -run TestFlowBenchmarkSuite ./internal/harness -v
```

Needs `CEREBRAS_API_KEY` and `TYPESAFE_API_KEY` in the repo-root `.env`. Each arm skips on its
own missing key and says so, so one vendor's absence does not take the run with it. Requests go
one at a time, because a benchmark reporting latency cannot also be saturating what it measures.

The table is printed and written, with every judgement, to
`internal/harness/testdata/flowbench-out/<timestamp>/`, which is gitignored. A run worth keeping
gets its table pasted in below.

## Results

Not yet run.
