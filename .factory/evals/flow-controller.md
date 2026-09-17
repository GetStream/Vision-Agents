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
| `gemma` | `gemma/gemma-4-26B-A4B-it`, our own deployment, given the production `flowInstructions` and `flowQuestion` and parsed by the production `parseFlow`. The incumbent, unaltered. `FLOW_BENCHMARK_GEMMA=cerebras/gemma-4-31b` puts the same set to Cerebras' public one instead |
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

Needs `GEMMA_BASE_URL` with `BASETEN_API_KEY`, and `TYPESAFE_API_KEY`, in the repo-root `.env`.
An arm whose model cannot be reached says so and stands down, so an undeployed Gemma or a
missing key does not take the run with it. Requests go one at a time, because a benchmark
reporting latency cannot also be saturating what it measures.

The table is printed and written, with every judgement, to
`internal/harness/testdata/flowbench-out/<timestamp>/`, which is gitignored. A run worth keeping
gets its table pasted in below.

## Results

Run 2026-09-17, 120 cases. The Gemma arm stood down: neither `GEMMA_BASE_URL` nor a Cerebras
key was set, so the incumbent column is still owed and nothing below is a comparison yet.

| Arm | Model | Correct | Floor free | Agent talking | Missed stop | False stop | Unreadable | p50 | p95 | $/1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jev-direct | `jev-latest` | 80.8% | 76.7% | 85.0% | 0 | 7 | 0 | 134ms | 490ms | $0.032 |
| jev-composed | `jev-latest` | 83.3% | 81.7% | 85.0% | 0 | 7 | 0 | 128ms | 527ms | $0.046 |

| State | Wants | jev-direct | jev-composed |
| --- | --- | --- | --- |
| `respond` | `answer` | 100% | 100% |
| `wait` | `wait` | 60% | 60% |
| `wait-digits` | `wait` | 50% | 60% |
| `wait-menu` | `wait` | 70% | 100% |
| `clarify` | `answer-clarify` | 80% | 70% |
| `ignore` | `ignore` | 100% | 100% |
| `stop` | `interrupt` | 100% | 100% |
| `shorten` | `shorten` | 20% | 20% |
| `continue-ack` | `continue` | 100% | 100% |
| `continue-noise` | `continue` | 100% | 100% |
| `continue-echo` | `continue` | 90% | 90% |
| `continue-elsewhere` | `ignore` | 100% | 100% |

Three things are worth saying before the incumbent arrives.

**Nothing was missed that mattered.** Zero missed stops across both arms: every caller who took
the floor got it. Both arms also read a cough, an acknowledgement, an echo and a room full of
other people correctly every time or nearly, which are four of the six states the agent has to
get right while it is talking.

**Shorten is where it falls down, and it falls down in the expensive direction.** Seven of ten
`shorten` cases came back as `interrupt`, and those seven are the entire false-stop count. A
caller who says "and put us on the patio" over the agent gets the answer abandoned rather than
cut short. That is a real cost — the agent loses the sentence it was halfway through and starts
again — but it is the cheaper of the two mistakes, and the distinction is genuinely fine: both
are additions to what was asked, and only the degree separates them. Whether Gemma draws the
line better is now the most interesting number in the benchmark.

**Composing helped exactly where it was meant to.** `wait-menu` went from 70% to 100%, because a
Noul asking "is this a recording reading out its options" is a question with one answer, whereas
folding it into a four-way choice makes it compete with `respond`. The composed arm is 43% dearer
per decision for 2.5 points overall, all of them in the wait family. Both arms are an order of
magnitude inside the 3s deadline, so latency is not what will decide this.

The confusion matrices and every judgement are in the run directory; the summary above is the
part worth keeping in the repository.
