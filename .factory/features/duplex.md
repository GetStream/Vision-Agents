# Speaking while listening

[Sprint 5](../sprint5.md), part D, and [Sprint 6](../sprint6.md). Every decision below is
taken in `converse` since [sprint 15](../sprint15.md); see
[the voice agent](voice-agent.md).

## Asked for

Instead of taking turns, the LLM should listen and talk at the same time, and know when to ask
clarifying questions.

## What exists

[cadence.go](../../acceleration/internal/agent/cadence.go) holds evolving transcript state per
participant. A revision that stays unchanged is sent to a second fast-model session in
[flow.go](../../acceleration/internal/harness/flow.go), which chooses `wait`, `ignore`,
`respond`, or `clarify`. Provider turn-start and turn-end events are no longer in the STT
contract, and a final transcript is not what triggers a response.

### Backchannels

While speech is evolving or delegated work is pending, a long gap since the agent last spoke
produces one short token straight through TTS. It never reaches the model and an otherwise idle
call stays quiet. `-backchannel` defaults on; `-backchannel=false` disables it.

### Overlap

When speech arrives over an answer, the controller chooses:

- `stop`: interrupt the current model and voice
- `shorten`: stop generating, but finish audio already sent to the voice
- `continue`: keep the floor and retain the new transcript for the next response

The controller is asked while the caller is still talking, on transcript revisions rather
than only once the words settle, so the floor can be given up mid-utterance. These
provisional asks (`over-` candidates) are bounded per utterance and never answer: the words
settle on their own and are answered through the ordinary path. The controller is told what
the agent has said so far, so it can tell a correction from the caller's line echoing back.

The reverse also holds: an unprompted turn the agent owes — a tool result, delegated work
coming back, a turn queued behind the reply — waits (`hold`) while a caller has evolving
words younger than `callerHold`, rather than taking the floor from somebody who has it.

Background speech is ignored using Stream participant identity and conversational addressee
classification.

### Clarifying questions

Three sources are routed through the [harness](harness.md):

- A task result carrying `Question`, which is the subagent saying it needs something only the
  caller knows.
- A transcript whose confidence is below `-min-confidence`. The agent checks what was
  meant instead of confidently answering the wrong question.
- A `clarify` flow decision when the words are clear but the intent is ambiguous.

## Observability

`Backchannel` and `OverlapDecided` join the agent's events, alongside `Delegated`,
`TaskSettled`, `TaskCancelled`, and `ConversationCompacted`. Each of these also produces a
`Decided`, which is the one stream that has all of them in the order they happened.

## Not done

Two voices mixed onto one participant track are classified from transcript context; there is no
biometric diarization or source separation.
