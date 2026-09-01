# Simulate and test

[Sprint 16](../sprint16.md), "Simulate & Test".

## Asked for

Write down a conversation to have with an agent and something that has to be true at the end
of it: a name, a type (text or audio), what to ask over several turns, whether to try ten
different ways of asking, what to evaluate, which agent, and which model judges it. Then a
page in the dashboard that lists them, runs one, shows past results and logs every run.

## What exists

[internal/simulation](../../acceleration/internal/simulation) is the runner and
[store/simulations.go](../../acceleration/internal/store/simulations.go) is the three tables
behind it: a **simulation** is what is asked, a **run** is one press of Run, and a **case** is
one conversation. Variations off gives a run one case; expanding gives it ten.

```
POST /v1/agents/simulations             name, agent, scenario, assertion, variations, judge
POST /v1/agents/simulations/{id}/run    returns the run, not the answer
GET  /v1/agents/simulation-runs         the log, or one simulation's history
GET  /v1/agents/simulation-runs/{id}    the run with every transcript and ruling
POST /v1/agents/simulation-runs/{id}/cancel
```

## The caller is a model

The thing that makes a scenario more than a script. "After the order is handled, change your
mind" cannot be split into fixed turns in advance, because whether the order was handled is
something only the agent's reply can say. So the caller is a second routed model
([caller.go](../../acceleration/internal/simulation/caller.go)) given the scenario as a
brief: it reads what the agent said, decides the next thing to say, and decides when
everything it called about has been dealt with.

It also makes variations nearly free. Ten ways of asking is one completion that rewrites the
brief ten times ([variations.go](../../acceleration/internal/simulation/variations.go)),
keeping every fact and varying only the wording. The scenario as written is always the first
of them, so a run of ten still asks the question the customer actually wrote.

A third model rules on the finished conversation against the assertion
([judge.go](../../acceleration/internal/simulation/judge.go)), following the shape
[review.go](../../acceleration/internal/session/review.go) already used for calls. It answers
only the question it was given: an agent that was rude and still placed the order passes. A
ruling that cannot be read errors the case rather than passing it, because a conversation
nobody managed to judge has not been judged.

## Knowing when a turn is over

The hard part, and worth reading before changing it.

One thing said to an agent can earn several replies: the turn that called a tool, the turn
that read what it returned, the turn a subagent's finding was worth. A caller that took the
first `Responded` for its answer would talk over the rest. So `Say` waits for two things
together: `Session.Busy()`, which is new and reports whether the agent is generating,
speaking, delegating or owes a follow-up; and a quiet window, because in the moment between
one of those replies ending and the next beginning the agent is genuinely doing nothing.
`Busy` covers the long waits and the window covers the handovers. Nobody is waiting on a
simulation, so it waits the handover out rather than talking into it.

## Out loud

An audio simulation is the same loop with a different transport
([audio.go](../../acceleration/internal/simulation/audio.go)). The caller gets a voice from
the TTS router and ears from the STT router, and the two ends meet in
[internal/loopback](../../acceleration/internal/loopback) — the paced edge promoted out of
[conversation_test.go](../../acceleration/internal/agent/conversation_test.go), which is
still its regression test.

The caller hears the agent **through the agent's own voice, transcribed**, rather than
through the text the model wrote. That is what "run the full pipeline" means, and it is what
catches a reply that is right on paper and unintelligible in the air. Both are stored: the
transcript line is what was heard, and `intended` beside it is what the agent meant, so a
failure caused by the voice can be told from one caused by the answer.

`audio.Resample` is new and is the only piece of DSP in this: providers emit 24 kHz and the
call carries 16 kHz mono, and the two existing converters need ffmpeg or cgo.

## What it needs to run

Postgres, sessions and LLM routing. Audio also needs speech routing, and is refused rather
than quietly held in writing where there is none. Without a database a simulation cannot even
be written down, and the paths say which of the three is missing.

## Changed along the way

- **`Session.Busy()`** and `Agent.Busy()`, above.
- **`Spec.Edge`**, so a caller can hand in the loopback it is holding the other end of.
  `ManagerOptions.Edge` is a factory keyed off the spec and cannot hand a particular one back.
- **`Spec.NoReview`**, so ten conversations do not each pay for a post-call summary the judge
  has already superseded.
- **A text session's own half of the conversation was never recorded.** `agent.Heard` is only
  emitted from the transcriber, so a call in writing stored the agent's replies and none of
  the prompts, and its review read one side of a conversation. Fixed in `Session.remember`.
- **`llmrouter.Await` and `llm.Unfence`**, which were the fourth and fifth copies of a loop
  and a fence stripper.
- **Campaigns leaked a session per call**, closing the session rather than the manager entry.

## Not done

- **Nothing runs a simulation on its own.** No schedule, and no run-on-save when an agent
  config changes, which is the obvious next thing.
- **No pass/fail history over time.** Every run is stored, but nothing charts whether an
  agent is getting better or worse.
- **One router owns a run.** A process that dies mid-run leaves it to be written off at the
  next start rather than picked up, which is the same assumption [campaigns](campaigns.md)
  make about contacts, minus the claiming.
- **Audio has one voice and no room.** The loopback carries silence between utterances; the
  noise and bystander tracks the conversation suite uses are not offered to a simulation.
