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

Run 2026-09-17, 120 cases, Gemma three times each and Jev once.

| Arm | Model | Correct | Floor free | Agent talking | Missed stop | False stop | Unreadable | Flipped | p50 | p95 | $/1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma | `gemma/gemma-4-26B-A4B-it` | **85.0%** | 83.3% | 86.7% | 0/30 | 12/330 (3.6%) | 2 | 4.2% | 179ms | 443ms | $0.149 |
| jev-direct | `jev-latest` | 80.8% | 76.7% | 85.0% | 0/10 | 7/110 (6.4%) | 0 | n/a | 144ms | **266ms** | **$0.032** |
| jev-composed | `jev-latest` | 83.3% | 81.7% | 85.0% | 0/10 | 7/110 (6.4%) | 0 | n/a | **131ms** | 246ms | $0.046 |

| State | Wants | gemma | jev-direct | jev-composed |
| --- | --- | --- | --- | --- |
| `respond` | `answer` | 100% | 100% | 100% |
| `wait` | `wait` | **100%** | 50% | 60% |
| `wait-digits` | `wait` | **90%** | 50% | 50% |
| `wait-menu` | `wait` | 100% | 80% | 100% |
| `clarify` | `answer-clarify` | 30% | **80%** | **80%** |
| `ignore` | `ignore` | 80% | **100%** | **100%** |
| `stop` | `interrupt` | 100% | 100% | 100% |
| `shorten` | `shorten` | **60%** | 20% | 20% |
| `continue-ack` | `continue` | 87% | **100%** | **100%** |
| `continue-noise` | `continue` | 93% | **100%** | **100%** |
| `continue-echo` | `continue` | 80% | **90%** | **90%** |
| `continue-elsewhere` | `ignore` | 100% | 100% | 100% |

### Gemma wins by 1.7 points and the two models are nothing alike

The headline is close enough to be noise — 85.0% against 83.3% over 120 cases — and the state
table underneath it is not close at all. They fail in almost opposite places, so which is better
depends entirely on which mistake a call can afford.

**Gemma knows when somebody has not finished. Jev often does not.** `wait` 100% against 60%, and
`wait-digits` 90% against 50%. This is the largest gap in the benchmark and it is Gemma's. Half
the time Jev hears "it's four four eight" against a request for an eight digit member number and
answers it. In a call that is the agent talking over the middle of somebody's account number,
which then has to be asked for again.

**Jev knows when a request is ambiguous. Gemma barely does.** `clarify` 80% against 30%: Gemma
answered 21 of 30 ambiguous requests as though they were clear. Asked to "cancel it" with two
appointments live, it cancels one. That is worse than a clumsy turn, because the agent does the
wrong thing confidently and the caller has no signal that it guessed.

**Jev is steadier about everything that is not a request.** It reads an acknowledgement, a cough,
an echo and background chatter correctly every time or all but once, where Gemma drops 1 to 2
cases in each. None of those are expensive individually; together they are why its "agent
talking" column is level with Gemma's despite Gemma being much better at `shorten`.

**Neither missed a single stop.** Thirty judgements for Gemma and ten for Jev where the caller
took the floor, and all forty got it. This is the failure that makes people hang up, and on this
set neither model has it. Both do give the floor up when they should not, and there Gemma is
better per judgement: 3.6% against 6.4%. Jev's false stops are all the same mistake — seven of
ten `shorten` cases read as `interrupt` — so a caller adding "and put us on the patio" loses the
sentence the agent was halfway through instead of having it trimmed.

**Jev is 3.2x cheaper and tighter at the tail.** $0.032 against $0.149 per thousand decisions,
and a p95 of 266ms against 443ms. Both are far inside the 3s deadline, so latency does not decide
this, but the tail is the number that matters on a live path and it is Jev's. Gemma also produced
two answers that would not parse and flipped its own verdict on 4.2% of cases across three
samples; Jev returns a distribution, so there is nothing to flip.

**Composing helped exactly where it was meant to.** `wait-menu` went 80% to 100%, because a Noul
asking "is this a recording reading out its options" has one answer, whereas folded into a
four-way choice it competes with `respond`. That is 2.5 points overall for 43% more per decision,
and it closes about a third of the gap to Gemma without touching the `clarify` advantage.

### What this suggests

Nothing here says replace the incumbent. It says the two models are good at different halves of
the job, and that a controller built from both would beat either: Jev is the better judge of
whether words were meant for the agent and whether they are a clear request, and Gemma the better
judge of whether they have finished. Jev being an order of magnitude cheaper and answering with a
probability rather than a sample makes it the cheaper half to add.

The obvious next step is not a third arm but a fourth question: give the composed arm Gemma's
`still_growing` judgement — or Gemma's alone — and see whether 83.3% moves past 85% at a fifth of
the cost.

Two caveats. `router.yaml` currently routes `llm-flow` to `gemini/gemini-3.8-flash` rather than to
either Gemma, so the incumbent measured here is not what a call runs through today; that arm is
worth adding before anything is decided. And 120 cases put a point or two of this inside the
noise, which is why the state table matters more than the headline.

The confusion matrices and every judgement are in the run directory.
