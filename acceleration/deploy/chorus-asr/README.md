# chorus-asr

chorus-asr is one speech model that hears the caller's last 16 s and answers the flow controller's
question from the audio: whether the caller has finished, what to do with the words
(respond, clarify, ignore), what to do about the agent's own speech (stop, shorten,
continue), and which of the agent's tools is wanted. The same server can write down what
was said. The router speaks to it as `internal/audiofloor`.

It plugs in twice, and each is optional:

- **Decisions.** The flow controller's `Hearer` (`internal/harness/hearer.go`), beside
  `Judge`, for an agent whose floor control opts in
  ([Floor control](#floor-control-floor_control)). A `Judge` is asked named questions about
  written state; this model is asked about the audio, in the words it was trained on, which
  only its server can write.
- **Transcripts.** The `stream/chorus-asr` transcriber, an `stt.STT` like any other,
  chosen by an agent that names it as its STT. With it a call runs no other transcriber.
  The model ends each utterance: the transcriber checks each pause from the voice's last
  frame, under an agent of its own when no agent acts on the model, and writes the
  utterance as one final when the model says the turn has ended
  ([Transcription alone](#transcription-alone)). When to answer stays the flow
  controller's.
- **Both, acting.** An agent that acts on the model and names it as its STT has the
  transcriber ask about each pause on its behalf, under its own instructions and tools,
  the decision and the words in one request: [Turns from the audio](#turns-from-the-audio).

## The loop's contract

What a caller can count on from an agent that acts on chorus-asr with `stream/chorus-asr` as
its STT. Each line is held by checks of the replay ([Test it](#test-it)), named in brackets:
`TestAudioFloorReplay` by number, `TestAudioFloorScenarios` by S, `TestAudioFloorContract` by
C. The loop acts on the model's rulings and nothing else: where the model rules wrong, the
check fails and the failure is the model's, reported as such, not worked around in the router.

**While the agent talks**, the caller is ruled on 350 ms voiced into what they say and every
`interval_ms` while they go on (the words so far), and at their first pause (the words said).

- **Ignored.** A backchannel ("uh-huh", "mm-hmm", "hm hm", "okay", "right", "yeah"), a laugh,
  a cough, the agent's own voice heard back on the line, somebody in the room speaking
  another language: the reply plays whole, nothing is shortened or stopped, and nothing is
  answered afterwards. [1, S7, C1]
- **Stopped, then quiet.** "Stop", "wait", "hold on": the agent stops within 800 ms of the
  caller's first word, none of what it was saying is heard after, and it says nothing more.
  A stop ruled over the agent, while the caller is still talking (its disposition is `wait`)
  or at their pause, is not a turn: the words it was ruled on are spent on it. [C2]
- **Stopped, all of it.** A stop silences everything the agent is saying or about to say. The
  agent takes a turn of its own, to say what a tool came back with, while the reply that called
  the tool is still playing: a stop over either stops both, the reply where the caller is in it
  and the turn behind it unsaid. A shortening stops both at its tail. [C2]
- **Stopped, then answered.** A correction or a new request ("no, Porto", "and in Lisbon?",
  "stop, what about Porto?" in one breath, or "stop" and then a question): the agent stops
  within 800 ms of the first word, and words the caller says after those the stop was ruled
  on make a turn, answered once. It is answered on all its words, the stop's too: a stop is
  often ruled on the start of the request it makes ("hi, what's the weather" over the
  greeting), which the words after it alone would cut ("like in Lisbon tomorrow?"). Where the
  model rules a request one more item ("and in Lisbon?" over a forecast), it is shortened, as
  below, and answered. [11, S2, S3, S10b, C3]
- **Stopped on its first words, then answered on all of them.** A correction that goes on
  after the stop ("wait, make it six, and one of us is gluten free instead" in one breath):
  the stop is ruled on its first words ("wait, make"), and once the agent has stopped, the
  checks that go on while the caller talks have no reply to stop, so they act on nothing and
  spend none of what they hear. The whole correction is answered once, the stop's words with
  it. A bare "stop" said and ended is still spent on the stop and not answered. [C3]
- **Gone on with, then answered.** A stop spends only words said to stop the agent: words the
  caller began once they had heard some of its answer to their turn (the reply, or what it
  went on to say of a tool's result). A caller who went on with their own turn before any of
  the answer went out, the agent having taken a pause inside it for the turn's end ("the name
  for the booking is Alvarez", begun 0.7 s before the reply to the request before it), is
  stopping nothing they heard: a stop ruled on what they say is acted on, and their words are
  their turn, answered with the part before them in the conversation, as the model rules on
  them at their end.
- **Shortened.** One more item for the request being answered ("and in Porto?" over a
  forecast): the agent gives way within 800 ms of the first word, as it would to a stop, the
  model ruling on the same checks: it adds nothing more to the reply, stops speaking within
  250 ms of the ruling, and answers the item. The model was trained on `shorten` as "one more item of the
  same kind for the request being answered, neither correcting nor replacing it", in a demo
  that let at most 250 ms of the audio playing finish, as its voice had no word timing; the
  router's voices have none either. A line the agent says of its own accord, a greeting, a
  check-in or an update, answers no request, so `shorten` over it stops it. [S4b, S8, C4]

**After an interruption** the floor is the caller's: their turn ends once, by the agent's
`max_wait_ms` point at its check times from the voice's last frame, and is answered on that
check's ruling, with no second wait. [S2, C5, C7]

**While the agent listens**, pauses inside a sentence, slow counts and dictation are waited
through as the agent's `max_wait_ms` point was measured to (the table below), and a finished
turn is answered at that point. The agent says nothing of its own accord, such as a check-in,
while the model is waiting on the caller to finish. [4, 5, S1a, S6, C6]

**Waits.** Every wait from the caller's voice to the reply is the policy's (first check,
interval, bars, cap), the server's time to rule and write the words, or the reply model's. A
final the model ended is put to the conversation at once, with no settle gap, no grace after
an overlap and no retry. [8, C5, C7]

What the loop can't know: the model's words and rulings are all it has. A stop ruled on
misheard words ("let's stop" for "don't stop") is a stop. Whether a stop's words need an answer
is not among its rulings: over the agent `stop` is "a correction, a different request, a
question, or a direct interruption" alike, and at the turn's end, with the floor free, the
server shows the model no conversation (`conversation_on_free_floor: false`), so it rules a
bare "Stop.", "Can you stop?" or "No, stop." `respond` as it does "the name for the booking is
Alvarez" (at 1,100 ms of pause, 0.79, 1.00 and 0.93 of the answers that end the turn, against
1.00). The loop tells them apart by when the caller began, against the agent's answer: a stop
begun before any of it went out spends nothing, and is answered as the model rules; a
continuation begun once some of the answer was heard, with a stop ruled on all of it, is spent.
An utterance begins where the speech gate opens, so a bare stop in a room loud enough to hold
the gate open from before the reply went out is taken as begun then, and answered. A stop first
ruled at a pause over the agent, on words the caller has finished ("has just said"), is acted
on as one ruled while they talk, and spent on its words alike: a "stop" short enough to end at
a pause before the first check over the agent (under 350 ms voiced), or one whose first words
that check rightly let pass ("okay, stop"). Words the model rules `continue` at a pause over
the agent, as it ruled "I hope that's all right." (0.51, where with the floor free it rules
`respond` 1.00), end as `ignore`, and are not answered: at that pause if the caller began them
over the agent, and at the cap if they began them before hearing it and the agent talks on
through it. The server writes no words of an utterance of four words or fewer, heard whole,
that it rules `continue` or `ignore` over the agent (a murmur, "uh-huh", a piece of the
agent's echo), and says why (`ignored`). Words a caller began over the agent that are ruled so
are lost unless a check over the agent without a pause wrote them, and those are kept only
where the server gave no reason for writing none at the pause: where it said the pause was a
voice in another language or a murmur, the check over the agent heard less of it, and may
have taken a room's voice for words in the caller's language.

## Configure

The router makes chorus-asr available, and each agent opts in to it.

| Variable | Purpose |
| --- | --- |
| `ROUTER_CHORUS_ASR_URL` | The server, such as `http://floor:8080`, with the path of a gateway it sits behind if it has one ([Behind a gateway](#behind-a-gateway)). Makes chorus-asr available: the `stream/chorus-asr` transcriber, and floor control for an agent that opts in. It rules no agent's floor of its own accord |
| `ROUTER_CHORUS_ASR_API_KEY` | Optional. Sent as `Authorization: Bearer <key>` on every request, for a server behind a gateway that asks for a key. Never logged, and taken out of what a gateway says back. Send it over `https` |
| `ROUTER_CHORUS_ASR_AUDIO` | `opus` (default), or `pcm`, 16-bit PCM at 8 to 20 times the bytes, to tell the codec apart from the model when debugging a decision |
| `ROUTER_CHORUS_ASR_INTERIMS` | `on` (default), which has the transcriber write the words so far while the caller talks ([Live words](#live-words)); or `off`, for a server short of transcription lanes, where the words come at the turn's end |
| `ROUTER_CHORUS_ASR_OVERLAP_TIMEOUT` | How long the router waits for a ruling while the caller talks over the agent, as a duration: `250ms` (default) up to `450ms`, and at least `100ms`. The router refuses to start outside that. A ruling that comes later is not acted on, and the flow controller rules instead ([Fallbacks](#fallbacks)). The checks of a pause keep their 1 s whatever it says. Set it from the server's round trip ([The wait over the agent](#the-wait-over-the-agent)) |

`cmd/router` and `cmd/agent` read these the same way (`audiofloor.Open`); `cmd/agent`'s one
agent opts in with `-floor-control stream/chorus-asr`.

`ROUTER_CHORUS_ASR_MODE`, which once had every agent on a router shadow or act on chorus-asr,
is no longer read. A router started with it, as a variable or in its settings file, acts on it
for nothing and warns once, at start, naming `floor_control`.

The server is started with its model and the `policy.json` that ships with that model.
`GET /v1/info` hands the policy to the router, so nothing about a checkpoint is in the
router. A server without one answers 501, and the router never acts on it.

Three ways to use it, each a short Go program in
[examples/routers/chorus_asr_example](../../../examples/routers/chorus_asr_example/README.md):

| Use | Router | Agent |
| --- | --- | --- |
| [Decisions and transcripts](../../../examples/routers/chorus_asr_example/decide_and_transcribe/main.go): one model hears the caller, writes the words and ends the turn | `ROUTER_CHORUS_ASR_URL` | `floor_control: {model: stream/chorus-asr}`, `stt: stream/chorus-asr`, its own `tts` |
| [Decisions only](../../../examples/routers/chorus_asr_example/decide_only/main.go): another transcriber writes the words | `ROUTER_CHORUS_ASR_URL` | `floor_control: {model: stream/chorus-asr}`, any `stt` |
| [Transcription only](../../../examples/routers/chorus_asr_example/transcribe_only/main.go): the words, through the router's STT | `ROUTER_CHORUS_ASR_URL` | a router config naming `stream/chorus-asr` |

Languages are optional. An agent's language hints, when it has any, are the languages its
callers speak: the server is told them when the agent is registered, and asked for the words
so far in them, so that it can drop a room's voice in another language. With none, the router
sends none and invents none: the agent is registered and ruled on all the same, its pauses
asked about and its words written in whatever language the server heard, with nothing
detected first. A tag the server's model does not speak is dropped by the server, not refused,
and no agent is turned away from floor control over its languages.

### The wait over the agent

The agent stops within 800 ms of the caller's first word over it ([The loop's
contract](#the-loops-contract)). The first check over it is asked once the caller has been
voiced for 350 ms, so the router can wait about 400 ms for that check's ruling and still keep
that line. It waits `ROUTER_CHORUS_ASR_OVERLAP_TIMEOUT` for every decision over the agent,
whether the transcriber asks for it or the flow controller does. The round trips measured for
one decision, p50 / p90:

| Server | Timed | One call | Four calls at once |
| --- | --- | --- | --- |
| TPU v6e, in the cluster | the server's decision | 11 / 14 ms | 44 / 100 ms |
| L4, in the cluster (staging) | the round trip | 109 / 118 ms | 376 / 495 ms |
| RTX PRO 6000 behind Baseten's gateway | end to end (the server's decision) | 237 / 255 ms (41 / 54 ms) | not measured |

- `250ms`, the default, for a TPU in the cluster.
- `400ms` for an L4, or for a server behind a gateway such as Baseten.
- A server slower than that needs to be made faster, not waited for: past about 400 ms the
  agent misses the 800 ms stop line anyway. An L4 with four calls at once is one: at
  `400ms` it still misses at least one ruling in ten.

`router_chorus_asr_round_trip_seconds` times every request for a decision or for words, by
`kind` (`over_agent`, `pause` or `transcribe`) and `outcome` (`ok`, `deadline`, `canceled`
for newer words or the call's end, or `error`), and is what to set the wait from.
`router_chorus_asr_deadline_misses_total` counts the requests the router stopped waiting
for, by `kind`, and each is logged at debug level with its `kind`, `deadline_ms` and
`took_ms`.

## Floor control: `floor_control`

Whether chorus-asr controls an agent's floor, when a caller has finished and what to do about
speech over the agent, is the app's to choose, per agent, beside its `stt`, `llm` and `tts`.
`floor_control` is one setting: `model` names the model that controls the floor, as `stt` names
the transcriber, and `max_wait_ms` is its one knob, how long a caller may wait once they stop
([How long a caller waits](#how-long-a-caller-waits-max_wait_ms)). An agent that names none
keeps the default flow controller, as every agent did before there was a choice:

```yaml
# agent.yaml
floor_control:
  model: stream/chorus-asr
  max_wait_ms: 1100   # optional: how long a caller may wait once they stop; default is the server's default point
```

With `stream/chorus-asr` the agent asks chorus-asr first and acts on its ruling, and the flow
controller rules a turn it does not answer in time. With `stt: stream/chorus-asr` too, this is
[the loop's contract](#the-loops-contract). With none, the flow controller rules the floor as
it does with no such model; chorus-asr still transcribes for an agent whose `stt` is
`stream/chorus-asr` ([Transcription alone](#transcription-alone)).

- **Where it is chosen.** A sync (`POST /v1/agents/sync`, as every SDK's folder sync sends
  it), a config's create, replace and patch, and a server-side `createSession` or the session
  socket's start frame, which choose for one session over its config's, all take the same
  field. Sent, it replaces the stored choice whole, `max_wait_ms` included, and
  `floor_control: {}` is the default flow controller again; left out, or null, the stored one
  stays. A device may not choose a session's (403).
- **What is refused.** `stream/chorus-asr` sent to a router without `ROUTER_CHORUS_ASR_URL` is
  refused at write time (400, `not_configured`), before anything is stored, saying what the
  router needs. So is, with a 400, a `max_wait_ms` without a `model`, a model nobody knows, a
  key it doesn't have, the string `floor_control` once was, and `end_of_turn`, which
  `max_wait_ms` replaced. A choice stored on one router and read on another without a server
  asks nothing there, and the router logs a warning for the session.
- **One router, many choices.** Each session runs by its own: on one router, an agent whose
  floor chorus-asr controls and one with the default flow controller take their calls side by
  side, and share nothing but the server. The Hearer, the transcriber's pause checks and the
  loop's contract are each session's, and an agent with none chosen behaves exactly as it did
  before.
- **A session row** keeps the floor control it was opened with, which reopening it keeps; one
  opened with none takes its config's.

## How long a caller waits: `max_wait_ms`

How long chorus-asr waits before it takes a pause for the end of a turn is the agent's
`floor_control.max_wait_ms`: the most a caller who has finished may wait for the answer.
Waiting longer cuts the caller off less often, and answers later. Left out, an agent ends its
turns by the server's `default` point.

The server measured a frontier of points on held-out calls, and these are the points of the
checkpoint this was written against. Each point's bars apply from `from_ms` into a pause, and
the turn has ended once as many checks of the pause in a row as its "In a row" have reached
their bar:

| Point | Chosen by | Bars | In a row | Cut-ins | Answers at (median, p90) | Real dictation pauses cut | Slow counts cut | Dictation guard |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 500 ms | `max_wait_ms` 800 to 1,099 | 0.8 from 0 ms, 0.3 from 500 ms | 1 | 14.5% | 500 ms, 800 ms | 16.0% | 9 of 9 | fails |
| 1,100 ms | `max_wait_ms` 1,100 to 1,399, and the default | 0.995 from 0 ms, 0.2 from 1,100 ms | 1 | 4.5% | 1,100 ms, 1,100 ms | 12.6% | 0 of 9 | passes |
| 1,400 ms | `max_wait_ms` 1,400 and over | 0.99 from 0 ms, 0.2 from 1,100 ms | 2 | 3.1% | 1,400 ms, 1,400 ms | 10.6% | 0 of 9 | passes |

Cut-ins are the share of mid-turn pauses a point ended a turn in. Real dictation pauses are a
caller's pauses inside a number, an ID or a spelled name; slow counts are counts read one
number a second, cut when the turn ended before their last number. A point passes the dictation
guard when it cuts at most 16.2% of the dictation pauses and no slow count. An agent whose
callers read out numbers or IDs, or spell names, wants a point that passes.

- **How a wait chooses.** `max_wait_ms` takes the point with the fewest cut-ins whose measured
  p90 wait is within it. That is the measured wait, not the one a point was fitted to: the
  500 ms point answers in 800 ms at p90, so a wait of 800 ms gets it, and a wait under 800 ms
  gets nothing measured and is refused.
- **What is checked, and when.** A write that chooses a wait is refused (400), before anything
  is stored, when it is shorter than every measured point's p90 wait, saying what the quickest
  is; it is checked against what the server publishes then. A sync that chooses a point
  failing the dictation guard is stored, and its `warnings` say what the point does to slow
  counts and dictation pauses. When the server can't say what it publishes, the wait is kept
  and the sync's `warnings` say so.
- **What it replaces.** A point brings its bars and its checks in a row. When a pause is
  checked, how much more a turn's first pause asks, and the cap stay the server's, which are
  fitted to its model. A run of checks is one pause's, one after another: the caller speaking,
  or a check under its bar, starts it again. A check is asked once the one before has
  answered, so a slow answer makes the next check late; it still counts, as the caller was
  quiet the while.
- **When it is read.** At each check, against the policy of the model the server runs then.
  A wait that model has no point within, as after a new checkpoint, ends the session's turns
  by its default, and a warning says so once per session.
- **What you see.** Once a session, and again if it changes, the router logs the point its
  turns end by: `end_of_turn`, `checks_in_a_row`, `dictation_guard`, and the point's
  `max_wait_ms`, `cut_ins`, `answer_ms_p90`, `dictation_pauses_cut`, `slow_counts_cut` and
  `slow_counts`. `end_of_turn` is `max_wait_ms:N` for the point fitted to `N` (whatever wait
  the agent asked for), or `preset:NAME` for the server's default.

### Turns from the audio

Acting (`floor_control: stream/chorus-asr`) with `stream/chorus-asr` as the agent's STT, the
session's point is applied as it was measured, at the server's check times from the voice's
last frame, and a pause costs one request:

- **Pauses from the audio.** The transcriber's speech gate (-48 dBFS and the speech
  detector, below) says when the voice stopped. The transcriber asks about the pause
  `first_check_ms` (200 ms) after the voice's last frame, then every `interval_ms`, each
  once the one before has answered, on the agent's behalf: the moment of the call is the
  one the flow controller would put, and each check is read by the session's Hearer, its
  point's bars and checks in a row (`audiofloor.Asker`, which the agent hands its
  transcriber when it opens it).
- **One request a pause.** The first check of a pause is `POST /v1/decide?agent=ID&transcript=true`
  with the whole utterance: the server decides on its last 16 s, answers that at once, and
  writes the words of all of it in the same pass. The rechecks ask for the decision alone,
  on what a decision hears. The agent is registered with the session's languages, for the
  words.
- **One final.** A check that ends the turn hands its ruling to the agent, and its words
  go out as the utterance's one final, which the conversation takes at once: the policy has
  had its wait, and the words never change after it. The flow controller's Hearer rules on
  those words as the check did, with no request of its own, and the reply is asked for at
  once. Over the agent a pause check says what to do about the floor and not whether the
  caller has finished: the model's answer there has no p_eot. Whose pause it is says what
  its ruling means, by when the caller began the utterance, against what they had heard of
  the agent (as a stop's words are spent, above):
  - *Begun before they had heard it*, the caller was going on with their own turn, which
    the agent began to answer over. A ruling that the agent may talk on says nothing of
    whether they have finished, so while the agent holds the floor the check ends nothing
    before the policy's `cap_ms`, where the turn has ended whatever the model says, and its
    ruling stands while the agent talks on, with nothing more asked. Ended at once, a 200 ms
    pause at the comma of "Last four 9821, 14 Cedar Lane", begun 2 s before the reply could
    be heard, ended the utterance on its first half, its words written as nothing.
  - *Begun over the agent*, while they could hear it, a ruling that it may talk on ends the
    utterance at that check, as the model let the agent talk through it: a murmur, a cough,
    its own voice heard back. What the caller says next is an utterance of its own; run on,
    the agent's echo took the caller's question in with it.

  Either way a `stop` or `shorten` is acted on at once, with the pause's words, as over the
  agent without a pause (below), and the ruling stands. Once the agent is quiet the pause is
  asked about at once with the floor free, and the point's bars end the turn, its checks in a
  row counted from the pause's first with the floor free. Once the agent has given the floor
  up, stopped or playing out the tail of a reply it shortened, a ruling at the caller's pause
  that they take it (`stop` or `shorten`, which a pause rules `respond`) ends their turn, as
  the model rules on it with the conversation in view and with the floor free it has none:
  "And in Lisbon." over a forecast it shortened is answered at the pause's first check, where
  the checks with the floor free waited on it for 2 s (p_eot 0.04 to 0.23). A ruling that the
  agent go on ends nothing then. A turn a check ended is over whatever comes next: speech
  before its words are written, the caller going on or somebody else in the room, is the
  next utterance's. Its words are the first check of its last pause's; where those come back
  empty and a check over the agent wrote some of the utterance, they are that check's, as far
  as it heard, and an utterance with no words at all sends nothing, logged as lost where the
  model ruled them to be answered and the server gave no reason for writing none.
- **Over the agent, without a pause.** While the agent speaks, a caller who has been voiced
  for 350 ms in the utterance is asked about without waiting for a pause, the decision and
  the words of the utterance so far (up to 16 s) in one request, then every `interval_ms`
  while they talk on, each once the one before has answered. A pause may never come over
  the agent: its own echo on the caller's line or the room keeps the speech gate open, and a
  caller who talks straight over it leaves none `first_check_ms` would catch, so the pause
  checks alone left the agent talking until its reply ended. A ruling to `stop` or `shorten`
  is acted on at once, as a provisional ruling about another transcriber's revision is
  (`OverlapDecided`, then the interruption): the agent drops its own echoed voice from the
  check's words first, as from any chorus-asr transcript, and an echo alone or a murmur
  stops nothing. The checks go on every `interval_ms` for as long as the caller's line is
  voiced over the agent, its own echo included (Limits). A stop ruled while the caller talks on
  is no turn: the words it was ruled on are spent on it, if the caller began them once some of
  the agent's answer had gone out (The loop's contract). The utterance runs on, uncut, and
  ends at its pauses with the floor free, by the agent's point, as the caller may go on
  after a pause; its final is heard and not answered if it says nothing more than the stop's
  words, and is the caller's turn if it does. A check that answers once its utterance has ended, a pause check
  having ruled on it, is not acted on. Shortened, the utterance goes on, its words shown as interims, and ends at
  a pause, where the item is answered. 350 ms is most of an "uh-huh", so the model rules on a
  murmur, which it was trained to let the agent talk through, rather than on the first
  syllable of anything, and a word that stops the agent, "no" or "stop", is said inside it.
- **A pause the gate can't hear.** A room (a TV, a street, people talking) can keep the
  speech gate voiced after the caller has finished, so that no pause begins, or one begins
  again at every burst of the room, and the turn waits for the caller to speak again. The
  level can't tell such a room from the caller, nor can the detector, but the words can: the
  server writes nothing of the room. With the floor free, every request that writes the
  utterance's words (a pause's first check, an interim) moves a clock, and while the gate
  holds the line voiced and nothing has written the words for 1.1 s of speech, the
  transcriber asks a held check, a pause's first check in all but the pause. A held check
  whose words, not none, are the same as they have been for 1.1 s, or the agent's point's
  `max_wait_ms` if longer, with the gate voiced for at least half of that time, ends the turn
  on its own ruling, as `cap_ms` ends one: the model's disposition and floor, whatever p_eot
  says. It is logged on the `chorus-asr checked` line with `held=true`, `still_ms` and, when
  it ended nothing, `why_not`, and counted in `router_chorus_asr_still_ends_total`. A quiet
  line is as it was: after the caller the gate is quiet and the pause's checks rule; a held
  check is asked there only in a voiced stretch with no pause and no interim for 1.1 s, and
  words that change end nothing.
  Nothing is held over the agent, nor for a transcriber of its own
  ([Transcription alone](#transcription-alone)). Words a room's voice is written as (a TV
  read as the caller) keep the clock going, as they keep any transcriber's words changing.
- **Live words, acted on at the end.** While the caller talks the transcriber shows the
  words so far, as interims ([Live words](#live-words)), and the agent shows them and acts
  on none: nothing supersedes a ruling or interrupts a reply about audio already ruled on.
  Over the agent the checks write the words, and it is their ruling that is acted on.
  Speech after a pause the model waited on begins another: its first check writes the whole
  utterance again.

With another transcriber a point is applied to the words as they hold still (the first
check once they have held still for the cadence's 350 ms gap). With `stream/chorus-asr`
transcribing and no agent acting on the model, with the default flow controller, the
transcriber checks the pauses all the same, under an agent of its own
([Transcription alone](#transcription-alone)), and the flow controller takes the final as
it takes any transcriber's. A check that fails, or no check ending the pause within 1 s
past `cap_ms`, leaves that pause to the words alone and the flow controller, as before. A
check that ended the turn is waited on for its words however long they take, which on a
CPU is seconds for a long turn.

### What a choice does to a call

The acceptance replay (`TestAudioFloorReplay`, [Test it](#test-it)) plays recorded calls
into a real agent in real time, chorus-asr transcribing and acting, with the flow
controller's language model answering wrong on purpose, so that a check passes only on the
model's rulings; `CHORUS_ASR_END_OF_TURN` chooses the calls' point, such as `{"max_wait_ms":1400}`. Against one CPU
server built from the policy above (f16, 2 deciding lanes and 1 transcribing), on a machine
shared with other work, so the times are indicative:

| Check | Default (the 1,100 ms point) | `max_wait_ms: 800` (the 500 ms point) | `max_wait_ms: 1400` |
| --- | --- | --- | --- |
| 1. "Uh-huh" over the reply | reply whole, no new answer | the same | the same |
| 2. One final after 10 s of silence, room tone, line noise at -50 dBFS | one final, one request for the words | the same | the same |
| 2. The reply echoed at -20 and -30 dB acting, -20 dB transcribing alone | the conversation as with no echo; the transcriber still writes the echo, and transcribing alone as a final of its own, the caller's final written within a check or two of their voice | the same | the same |
| 3. A breath or a burst of noise after a turn | no other script | the same | the same |
| 4. One to ten, a second apart | answered after "ten" | answered "One." 1.8 s in, as the point's 9 of 9 slow counts cut says | answered after "ten" |
| 5. 19.8 s with pauses under a second | one final with its beginning, one answer | the same | the same |
| 6. A cough alone | nothing | the same | the same |
| 7. A new agent registers | 399 ms, inside the 1 s | 233 ms | 178 ms |
| 9. Transcribing alone (floor control off) | one final within a check or two of the voice, the decision and the words in one request under the transcriber's own agent; no words asked for alone | the same | the same |

From the end of the caller's voice: "Will it rain in Porto this weekend?" (five calls), "Hi,
what's the weather like in Lisbon tomorrow?" (two: a new agent's first turn, then one that
pauses after "Hi"), the slow count's "ten", and the 19.8 s turn:

| | Default (the 1,100 ms point) | `max_wait_ms: 800` (the 500 ms point) | `max_wait_ms: 1400` |
| --- | --- | --- | --- |
| The server's ruling, rain | 303 to 328 ms | 290 to 335 ms | 605 to 628 ms: the second check in a row |
| The turn answered, rain | 680 to 814 ms | 589 to 853 ms | 666 to 689 ms |
| The turn answered, weather | 1.30 and 1.11 s | 1.14 and 1.00 s | 0.88 and 0.79 s |
| The turn answered, slow count | 2.05 s | cut after "One." | 2.53 s |
| The turn answered, 19.8 s turn | 4.08 s | 2.47 s | 3.14 s |

The turn is answered once the ruling and the first check's words are both in: the words of
2 s of speech take about 400 ms on this CPU, so a point that rules on its second check loses
little to one that rules on its first: the 1,400 ms point, which waits for two checks in a
row, answered these questions no later than the 1,100 ms one. Both
questions reach p_eot over 0.995 at the first check, so `safe` and `fast` rule on them alike.
A breath after a question restarts the pause, and the turn is answered 200 ms or more after
the breath.

Before turns from the audio, with the transcriber writing at pauses and the Hearer asked
350 ms after the words held still (against the server before its frontier, `safe`): "uh-huh"
stopped the reply, the decision taking 251 ms of the 250 a pause over the agent has; the
echoed reply was answered twice as the caller and interrupted three times; the slow count
was answered at "One." (p_eot 0.9986 against 0.995, about 0.7 s into the pause); and the
19.8 s turn was answered four times while the caller was still talking, as the words held
still for longer than `cap_ms` between pauses. Each pause cost two requests, the words and
then the decision; the rain question was answered 1.08 to 1.10 s after the voice. On a
person's recording of the weather question, "Hi." was a turn of its own and the question was
answered 1.07 and 1.19 s after the voice; with turns from the audio the pause after "Hi." was
waited on (p_eot 0.43 and 0.39 at 200 ms) and the question answered at 692 and 735 ms.

### Scenarios

What `TestAudioFloorScenarios` measured against the TPU server, acting under the `safe`
default, the reply echoed back at -30 dB where the caller speaks over it. Times are from the
end of the caller's voice unless said otherwise; over the agent they are from the caller's
first word, and the loop acts within the policy's 300 ms interval on the model's first
ruling against the reply whose words carry any of the caller's: one on the echo alone, which
the first check often is, stops nothing.

| Scenario | What the loop did |
| --- | --- |
| S1a. Two pauses of 800 ms inside a turn | One final, one answer, no cut-in: ruled 271 ms, answered 420 ms |
| S1b. Two pauses of 1,300 ms inside a turn | Cut in at both, three turns: past 1,100 ms the `safe` point's bar is 0.2, as measured (4.5% of mid-turn pauses) |
| S2a. "No, Porto." 500 ms into the reply | The model ruled `stop` 544 ms after the first word, on the check at 350 ms voiced, and the agent stopped at 568 ms; the utterance ran on uncut, and "No, Porto." was answered 1.19 s after the voice, at the 1,100 ms bar |
| S2b. "No, I meant Porto." with a 300 ms pause, 500 ms into the reply | Ruled `stop` 543 ms after the first word, stopped at 567 ms; one turn, answered 320 ms |
| S3. "Sorry, what was the second one?" 4 s into a list | Ruled `stop` 383 ms after the first word, stopped at 405 ms; answered 1.19 s after the voice |
| S4a. A second question 420 ms after the first, as the reply begins | The first reply cut, and the second question answered 1.75 s after its voice |
| S4b. "And in Porto?" 2 s into the reply | Heard as "and in Porto" out of the echo; the model ruled `shorten` 588 ms after the first word, the agent gave way at 652 ms and stopped speaking 250 ms on; answered 2.73 s after the voice, the free floor's bars waiting on "and in Porto" |
| S5. Silence after the agent asks a question | One answer; nothing written and no request made of the silence |
| S6. "So I was thinking [1.2 s] hmm [0.9 s] maybe Friday…" | One turn, no cut-in: ruled 264 ms, answered 402 ms |
| S7. A cough 2 s into the reply | The reply heard whole, nothing answered |
| S8. "And also, should I bring an umbrella…" 1 s into the reply | The model's first `shorten` with the caller's words in it came 642 ms after the first word, the agent gave way at 669 ms and stopped speaking 250 ms on; the words heard whole, without the echo, and answered 323 ms after the voice |
| S9a. Every decision 400 ms late | Ruled 662 ms, answered 766 ms, every turn the model's |
| S9b. 503 to the first two decisions | The first turn answered by the flow controller 726 ms after the voice; the next ruled by the model, 265 ms |
| S9c. 503 to every `POST /v1/transcribe` | 15 interims refused and the pauses' words shown instead; one final with the turn's beginning, answered 968 ms |
| S10a. The agent greets first, the caller waits | Answered 391 ms; the greeting's echo nothing heard |
| S10b. The caller asks 1 s into the greeting | The greeting stopped 1.34 s after the first word, on the model's ruling; the question answered whole, 322 ms after the voice |
| S11. An 80 s monologue | Three finals, 60.0 s of 210 words, 2.6 s and 17.9 s: the utterance ends at 60 s as it is and is answered then, 62 s in, while the caller talks on; the caller's next words stop that reply, and the rest is answered 81 s in |
| S12a. Portuguese after English, callers speak `en` | Nothing written, nothing answered: a language the callers don't speak is noise to the server |
| S12b. Portuguese after English, no languages | Written in Portuguese, live words and final, answered 383 ms |

### Live words

A caller counting slowly for 40 s saw nothing until the turn ended. In every mode the
transcriber now shows the words so far while the caller talks, as interim transcripts
(`replacement` of the same utterance, `stt.Transcript.Interim`), as other transcribers do:

- **The words of a pause the model waits on**, which its first check wrote anyway, go out
  as soon as they are written rather than kept for the turn's end. A caller who pauses
  between numbers sees each number as they say it, at no extra cost.
- **About once a second of speech with no pause in it**, the words so far are asked for
  alone (`POST /v1/transcribe`), the whole utterance so far, which a replacement must be:
  one at a time per participant, and one the server was too busy for (503, 429 or
  `transcript_error: busy`) or failed is not asked for again; the next is, a second of
  speech on. `router_chorus_asr_interims_total` counts each by what became of it:
  `written`, `late`, `unstable`, `empty`, `foreign`, `refused` or `failed`.
- **In the utterance's own language.** The server names the language it heard with every
  transcript (`language`). The words so far are asked for in the language of the
  utterance's first words at a pause, which the server had a whole phrase to tell, and
  words that come back in another, named otherwise or refused as one the callers don't
  speak (`noise: language`), are dropped (`foreign`), as are a pause's words in another
  language, which wait for the final. With languages declared the words so far are asked
  for in them until the utterance's language is known, and in it alone from then on. With
  none declared they are asked for in none, and the server writes what it heard: asked to
  guess the language of a second of speech, the server has guessed differently from one
  interim to the next over somebody else talking in the room, and an interim that switches
  script on the words shown is dropped (`unstable`). The final is the model's, in whatever
  language it heard the turn.
- **The line never goes back.** An interim holds up no pause check, no decision and no
  final: it runs on a goroutine of its own, and one that answers after the final, or after
  newer words, is dropped (`late`). An interim of the next utterance waits for the last
  one's final, which a slow server may still be writing. Once the words shown are stable,
  carried on by the interim after them, an interim that does not carry them on in turn is
  dropped (`unstable`): fewer words, or an opening and words that are not theirs, as two
  writings of the same audio over noise can be. Words shown once and never carried on may
  be anything, the agent's echo or the room, and the next words replace them. No interim
  brings a script the words shown are not in. The final has the last word, and the turn's
  end is exactly as it is without interims.
- **In an agent's call** interims are shown (`Hearing`) under the line the turn is going
  into and put to nothing: not the cadence, which would read words that hold still for
  350 ms as a pause, nor the flow controller, an overlap or a backchannel. The agent's own
  echo is dropped from them as from any chorus-asr transcript. Over the agent the words
  come with the checks that ask about the caller without a pause
  ([Turns from the audio](#turns-from-the-audio)), and it is the checks' ruling that is
  acted on. On the router's STT socket interims are replacements like any transcriber's.

**What it costs.** The words have no 16 s limit, so an interim of a long utterance would
grow with it. The audio an interim sends a second of speech is therefore bounded by what a
decision hears: under 16 s of utterance one is asked once a second, past it less often as
the utterance grows (every 2 s at 32 s, every 3 s at 48 s), so a caller who talks on sends
at most the words of 16 s a second however long they go on. On the TPU the words of 16 s
take 50 to 250 ms and those of 43 s about 430 ms, so a caller talking on without a pause
holds a transcription lane for at most a quarter of the time, and typically a tenth:

| Transcription lanes | Callers talking on at once before interims are refused |
| ---: | --- |
| 1 | 4 at 250 ms a second, 20 at 50 ms |
| 2 | 8 at 250 ms a second, 40 at 50 ms |

A caller who pauses costs less: the pause's first check writes the words and restarts the
second. Past that, interims are refused and dropped, and the requests that matter, the
pause's words and a final asked for alone, which are asked again, queue behind the
interims still running; a deployment short of lanes turns interims off with
`ROUTER_CHORUS_ASR_INTERIMS=off`, and the words come at the turn's end as they did.
Decisions run on their own lanes and are unaffected. Interims are not billed: only finals
are.

## Wire protocol

`GET /v1/info`: what the server serves, here with two of its three points left out:

```
{"template":"v3.5","conversation_on_free_floor":false,"end_of_turn":{"first_check_ms":200,"interval_ms":300,"cap_ms":2500,"first_pause_logit":0,"fitted_to":"...","measured_on":"...","default":"safe","presets":{"safe":1100,"fast":500},"frontier":[...,{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":1,"measured":{"cut_ins":0.0454,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.1257,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}},...]},"max_tools":21,"sample_rate":16000,"window_samples":256000}
```

`end_of_turn` is fitted to the server's model. A pause is checked `first_check_ms` into it
and every `interval_ms` after: counted from the voice's last frame when chorus-asr is the
transcriber too ([Turns from the audio](#turns-from-the-audio)), and with another
transcriber from when the words last changed, the first check once they have held still for
the cadence's 350 ms gap. At each check the bar is that of the last of the point's
`bars` to have begun, `from_ms` into the pause, raised by `first_pause_logit` in log-odds on
the turn's first pause, and the check reaches it when `p_eot` is at least the bar. The turn
has ended at the first check that reaches its bar with the `checks_in_a_row - 1` checks of
the pause before it, and at `cap_ms` of pause whatever the model says. `frontier` is the
measured points, `presets` name points by their `max_wait_ms`, and `default` is the preset
for an agent that chooses no `max_wait_ms` (see
[How long a caller waits](#how-long-a-caller-waits-max_wait_ms)). A point's `measured` is
what it did on held-out calls: `cut_ins`, `answer_ms_median`, `answer_ms_p90`,
`dictation_pauses_cut`, `slow_counts_cut` of `slow_counts`, and `dictation_guard`. A policy
without a frontier, a point the router can't run as written, or a preset naming no point is
one the router doesn't act on, as a server without a policy is. These values belong to one
checkpoint, and another one comes with its own.

`POST /v1/agents`: an agent, by what it was told and its customer tools. The router's own
tools (ending or transferring the call, pressing keys, the knowledge and web lookups,
confirming a tool call, video frames) are left out: listing them changes the model's
answers. The ID follows the content, so an agent registers once and again only when it
changes or the server restarts (a 404).

```
{"instructions":"You answer the phone for a weather service.","tools":[{"name":"get_weather","description":"Get the forecast for a city.","parameters":{"type":"object","required":["city"]}}]}
{"agent":"eb4fb3f2cb56bb977b7f0376680a4aa5","template":"v3.5","options":["respond","wait","clarify","ignore","cannot","get_weather"],"warmed_lanes":1,"warm_ms":67.85}
```

`POST /v1/decide?agent=ID`: a `multipart/form-data` body of a `turn` part, the moment of
the call as UTF-8 JSON, then an `audio` part: Ogg Opus (`audio/ogg`), or with
`ROUTER_CHORUS_ASR_AUDIO=pcm` 16 kHz mono PCM (`audio/pcm;rate=16000;channels=1;format=s16le`).

```
turn: {"conversation":[{"agent":true,"text":"Weather line, how can I help?"}]}

{"answer":"get_weather","answer_probability":0.9988813,"p_eot":0.9999969,"options":[{"name":"respond","p":0.00015754762},{"name":"wait","p":0.0000031085704},{"name":"clarify","p":0.00080397225},{"name":"ignore","p":0.0000036620925},{"name":"cannot","p":0.00015045748},{"name":"get_weather","p":0.9988813}],"sample_rate":16000,"samples":35364,"window_samples":35364,"decide_ms":102.45,"ruling":{"disposition":"respond","floor":"continue","tool":"get_weather"}}
```

`POST /v1/transcribe?languages=en`: a `multipart/form-data` body of one `audio` part, as a
decision's, and the languages the callers speak, the session's language hints. A server that
takes only a raw body here answers 415 ([Behind a gateway](#behind-a-gateway)). The answer is what was
said, of all of the audio, with no agent, decision or policy:

```
{"transcript":{"text":"Hi, what's the weather like in Lisbon tomorrow?","language":"English"},"transcript_ms":311.4}
```

The turn's fields are `speaking`, `said` (the agent's reply so far), `unfinished`,
`speaker` (`caller` or `another_voice`; a participant's name never leaves the router) and
`conversation`. They are what the model's rows were written with: `conversation` is what was
said on the call before the reply being spoken, the caller's words and the agent's, without
turns typed or handed in by the app or tools' results, and `said` is the reply as far as the
caller has heard it. The router has no word timing, so that is the time since the edge's track
took the reply's first audible frame at the voice's speaking rate on the call, from the
syntheses it finished whole, cut back to a whole word; before one has finished it sends the
reply as written. The router sends what a prompt shows: the reply's last 800 characters, the
lines of the last six turns, and each line's last 40 words. The turn goes in the body
because the longest, 800 characters of reply and six lines of 40 words, escapes to 10,285
bytes in Hindi and 12,035 in Russian as JSON in a header, past the 8 KiB a proxy allows one
header line.

### Behind a gateway

A server behind a gateway, such as a custom server on Baseten, is at the gateway's path, and
every route goes under it: `ROUTER_CHORUS_ASR_URL=https://model-<id>.api.baseten.co/environments/production/sync`
asks `…/sync/v1/decide`. `ROUTER_CHORUS_ASR_API_KEY` goes with every request as
`Authorization: Bearer <key>`. Baseten answers 401 without a key or with a wrong one, which
the router takes as misconfigured, not as a server down.

Measured on 2026-11-05 against one Baseten L4 (`L4:4x16`), through its sync path, from a
client in Europe about 120 ms from the gateway, with this router's client and Opus:

| | p50 | p99 | the server's own |
| --- | ---: | ---: | ---: |
| Decision, a 16 s window (28 KB) | 224 ms | 240 ms | 97 / 106 ms |
| Words with the decision, 16 s / 21.5 s / 43 s (15 each) | 1.11 / 1.63 / 1.87 s | 1.12 / 1.65 / 1.90 s | 0.88 / 1.29 / 1.52 s |
| `GET /v1/info` | 134 ms | 410 ms | |

- **The decision comes first.** The gateway passes each line as the server writes it: on a
  43 s turn the decision's line arrived at 373 ms (median of 10, HTTP/2) and the words at
  1.89 s, 1.51 s later, which is the server's `transcript_ms`. Over HTTP/1.1, and asking for
  no compression, the same.
- A decision line on a 65 KB turn came 120 ms, one round trip, after one on a 46 KB turn,
  as an HTTP/2 window of 64 KiB, which Baseten documents, would make it.
- The gateway takes up to 100 MiB of body, above the server's own bound, and its headers end
  between 8 and 16 KiB, past which it closes the connection (`GOAWAY`, `ENHANCE_YOUR_CALM`).
- Callers, each with an interim a second while talking and checks at 200, 500, 800 and
  1,100 ms of each pause, the first with the words: at 1, decisions at 132 ms p99 at the
  server and interims at 1.16 s; at 2, 135 ms and 1.18 s, 3% of interims turned away; at 3,
  184 ms. No count holds interims within 1 s, as on a cluster's L4. The interims here asked
  for a decision too, since `/v1/transcribe` didn't pass then, which costs the GPU more.
- A model scaled to zero starts on the first request, which waits for it at the gateway: 131 s
  to the first answer, most of it loading the model. The router's checks can't wait that long,
  so a router that acts on the model wants a replica kept up.
- **Every route sends its audio in a multipart body.** Behind Baseten, every raw body with an
  `audio/*` type, over HTTP/2 and HTTP/1.1, reached the server as a type it doesn't take, and
  it answered 415; multipart and JSON bodies passed. So `POST /v1/transcribe` sends its audio
  as a decision does, in a part named `audio` with its own type. That needs a chorus-asr
  server whose `/v1/transcribe` takes a multipart body: an older one answers it 415, behind a
  gateway or not, so its interims are not written, a transcriber whose words come from
  `/v1/transcribe` alone stops with a fatal error, and the acceptance replay stops in its
  warm-up. Its decisions, and the words a check asks for with `transcript=true`, still work.

## Transcription alone

The transcriber needs only `ROUTER_CHORUS_ASR_URL`. With its floor control off, as an
agent's is unless it opts in, an agent that names `stream/chorus-asr` as its `stt` gets the
model's words and the flow controller it would have had anyway, and so does a router config
naming it as an STT provider for a pipeline of your own:

```yaml
stt: stream/chorus-asr
```

The model still ends the turns. With no agent acting on it there is no prompt to decide
under, so the transcriber brings one of its own: a plain voice assistant with no tools and
no conversation, which names no business and asks for no behaviour, so that what it adds to
the decision is as near nothing as a prompt can be and `p_eot` is the audio's. It is
registered through `POST /v1/agents` like any other agent, once per server, and its turns
end by the server's `default` preset. The pauses are checked as they are acting
([Turns from the audio](#turns-from-the-audio)): the first check of a pause,
`first_check_ms` after the voice's last frame, is `POST /v1/decide?transcript=true`, the
decision and the words of the whole utterance in one request, and the utterance goes out
as one final when the model says the turn has ended, within a check or two of the voice
on a server that keeps up. Its ruling goes nowhere: what to do with the words is your
pipeline's, or the flow controller's, as with any transcriber.

Before this the transcriber ended an utterance at a pause of 2.5 s, the policy's cap, and
a pipeline that answered quickly had its own reply, heard back on the line, written into
the caller's utterance: "Will it rain in Porto this weekend? Tomorrow in Lisbon it will be
sunny…" as one final, 3.5 s after the caller stopped. On the replay against the TPU server,
transcribing alone under the `safe` default, the weather question reached a `p_eot` of
0.9955 at its first check and was a final 320 ms after the voice; the rain question sat at
0.965 to 0.975 over its first four checks, under the point's 0.995 bar, and ended at its
1,100 ms bar, a final 1.17 s after the voice, with the reply's echo a final of its own. An
agent's own prompt tells the model more: acting, the same question reaches 0.995 at the
first check ([What a choice does to a call](#what-a-choice-does-to-a-call)).

Decisions never hold the words up. A pause the server does not rule on, because the check
failed or none ended the pause within 1 s past `cap_ms`, falls to the words alone: the
utterance so far is written once the pause reaches 200 ms, asked for alone
(`POST /v1/transcribe`), and the utterance ends at a pause of 2.5 s with one final, as it
did before. A server started without a policy, which answers 501, is asked once a session
and transcribes all the same, the words alone from then on. The cap stays the fallback's
silence: with no model to say otherwise, a shorter silence would cut the pauses a caller
leaves inside a number or a spelled name, which the policy's points were fitted to leave
alone, and what a check decides it decides sooner than the cap anyway.

**Echo cancellation is the edge's.** The transcriber writes what is on the caller's line,
and a line with no acoustic echo cancellation carries the app's own voice back: a phone's
speaker into its microphone, or a browser without AEC. In an agent's call the router knows
what its agent said and drops the echo before the conversation sees it (Fallbacks, below).
Transcribing alone it does not know the app's voice, so the echo comes back as words, as it
does from any transcriber: a final of its own, after the caller's, rather than part of
theirs. Run AEC on the device or the media server (WebRTC's, or the SFU's), as every STT
over a loudspeaker needs.

### The audio

Each participant's audio is encoded once, as it arrives: one Opus encoder per participant
(gopus, VoIP, 32 kb/s, at the 16 kHz mono the router hears), fed 20 ms frames whatever size
the pieces come in, and a ring of the packets of the last 16 s. A decision sends the packets
it holds as an Ogg Opus stream of their own: OpusHead with a pre-skip of the encoder's
lookahead (312 at 48 kHz, 6.5 ms), OpusTags, then up to 255 lacing values to a page, granule
positions counted from the ring's first packet. A decoder that starts on a packet from the
middle of the encoder's output converges in its first frames, and the pre-skip lines the
audio up with the first frame's. What a decision hears ends up to a frame and 6.5 ms before
the newest sample, which at a pause of 200 ms or more is quiet.

## How a ruling maps onto the flow controller

The server says what, and the router says when. `ruling` is what to do once the turn has
ended; with the floor free the router has it ended when `p_eot` has reached the bar at as many
checks in a row as the agent's point asks, and otherwise rules wait and puts the same words
again after `interval_ms`. The
first check of a pause is the first time its words are put (`FlowTurn.Waited` false), the
turn's first pause is one before any wait, and the pause runs from the voice's last frame
when the transcriber is chorus-asr too, and otherwise from when the words last changed.

| Model | `Decided.Disposition` | `Decided.Floor` | Also |
| --- | --- | --- | --- |
| floor free, fewer checks in a row at the agent's bar than its point asks, and the pause under the cap | `wait` | `continue` | `Again`: `interval_ms` |
| `respond`, `clarify`, `ignore` | as said | `continue` | |
| `cannot`, or a tool | `respond` | `continue` | |
| over the agent: `stop`, `shorten` | `wait` while unfinished, else `respond` | as said | |
| over the agent: `continue`, `ignore` | `ignore` | `continue` | |

Nothing acts on the tool the caller wants, or on `cannot`, and a tool's approval is
untouched. Acting, a decision's `Ruler` is `heard`.

The words always count. Floor control says when the agent speaks, never whether what the
caller said reaches it. `ignore` with words, over the agent or not, says they were no turn
for the agent: a murmur ("yeah", "mm-hmm"), a voice in the room, a radio, or a caller's aside
("Yeah, zip is 46281." over a reply). They are heard, and kept for the next reply to the
caller, which is told they may be someone else and uses them only where they fit, a line the
agent says of its own accord included; the agent is not interrupted, and nothing is answered
on their own account: a caller's line the model took for background is answered with their
next turn, and the model's ignores are its training's to get right. A cough or a lone "uh"
stays ignored. With another transcriber writing the words,
the runs it had settled go out as a turn of their own when the floor steps aside or takes the
turns back, a final that comes after the floor gave up waiting for it is the next turn's (its
words past the ones the turn ended on), and words settled with no voice heard since are a
turn of their own 700 ms later: no final it writes is dropped.

## Fallbacks

- **Acting.** The model has a third of the flow controller's deadline for a pause, 1 s, and
  `ROUTER_CHORUS_ASR_OVERLAP_TIMEOUT` for words over the agent, 250 ms by default, a third of
  its 750 ms ([The wait over the agent](#the-wait-over-the-agent)). A timeout, an error, a
  refused agent (more than 21 customer tools), or a server without a policy leaves the turn
  to the flow controller, language model or judge, with the rest of its deadline. A timeout
  is counted in `router_chorus_asr_deadline_misses_total`. `router_chorus_asr_checks_total`
  counts each check by outcome; acting, every outcome but `answered` and `superseded` is a
  turn the flow controller decided.
- **Transcripts.** A pause no check ruled on falls to the words alone, check by check,
  and a server that can't rule at all (no policy, 501) is asked once a session. A server
  that is busy, restarting or slow costs a request, not the
  session: a 5xx, 404, 408 or 429, a timeout or a reset connection is logged and counted in
  `router_chorus_asr_transcribe_failures_total` by status, `timeout` or `connection`, and
  the utterance's words are asked for again, whole, at its end or its next pause after more
  speech. A server the router is misconfigured for (another 4xx, or no policy where one is
  needed) is fatal at once, and one that is down for the call is fatal after three requests
  for words in a row it did not answer at all (refused, reset, timed out, or a 5xx, 404 or
  408; not a 503 or 429, which is a server shedding load, nor words it answered empty), the
  last at least 8 s after the first, with none answered in between. Either hands the call to
  the next transcriber the target names, rather than leave it without words; a target that
  names this transcriber alone has no next one. Decisions are unaffected: a check the server
  does not answer falls to the flow controller, check by check, whatever became of the
  words. A request for words times out at 8 s and the audio's length again.
- **Echo.** A phone with no echo cancellation sends the agent's voice back down the
  caller's line, and the model writes it as the caller's words, or around theirs when they
  speak over it. The agent drops from a chorus-asr transcript every run of its own words,
  near enough (a quarter of them may be heard wrong, one at a time), with at least three of
  them heard right in a row as it said them, when the audio the words are of came while the
  agent's voice was going out, or began within 800 ms of its last audible sample going out,
  but for a run with two or more words of the caller's own before it and after it, more
  than a murmur, which they said in their sentence: what is left, before, between and after
  the runs, is the caller's, and an echo alone is nothing heard
  (`router_agent_chorus_asr_echoes_total`). When the voice goes out is the edge's pace:
  each chunk from when it was queued, after what was queued ahead of it, for as long as it
  plays, and its silences, the padding after the last word included, carry no words;
  speech the edge throws away on an interruption never goes out. 800 ms is the way out to
  the caller and back, each at most ITU-T G.114's 400 ms one-way planning limit. It does so
  in every mode, interims included, and a pause check's ruling is kept on the words so
  dropped, which are the words the turn is matched by; the transcriber itself still writes
  what it heard. Other transcribers are left as they are.
- **Noise.** The transcriber drops nothing itself. The server returns empty text for audio
  with no speech in it and for words in a language the callers don't speak, and cuts a
  fragment in a script they don't write from either end of the words. Empty text is nothing
  heard: it writes nothing, and an utterance ends on the last words that weren't empty.
- **An utterance's life.** A frame is speech when it is at least -48 dBFS and Opus's speech
  detector (gopus) scores it 0.5 or more: the level says when the voice stops, and the
  score tells speech from steady noise and hum. The transcriber asks for an utterance's
  words at a pause only when speech has come since it last did, so no audio is written
  twice; it ends the utterance with one final, and the next begins after it, with none of
  its audio.
- **A restart, or another model.** A 404 has the router register again. The policy it read
  from `/v1/info` is read again in the background, every 5 minutes and after a 404 or a 422
  refusing an agent, with the last one read standing until then: no decision waits on it,
  and a read that fails is tried again after 30 s. An agent written meanwhile has its
  `floor_control.max_wait_ms` checked against `/v1/info` as it answers then.

## Measured

All on an Apple M4 Max, built with `GOEXPERIMENT=simd`.

**The router's cost.** Benchmarks in `internal/audiofloor` (`BenchmarkAFrame`,
`BenchmarkADecisionsBody`), median of 6 to 8 runs, on 16 s of continuous speech and on 16 s
of a caller counting with pauses:

| | Before: PCM, copied from a ring at each decision | Now: Opus, encoded once |
| --- | --- | --- |
| Each 20 ms frame as it arrives | 14 ns | 123 µs (0.6% of a core per participant), nothing allocated |
| A decision's body, continuous speech | 512,476 bytes, 261 µs, 1.48 MiB allocated | 65,065 bytes, 205 µs, 284 KiB |
| A decision's body, speech with pauses | 512,476 bytes, 255 µs, 1.48 MiB | 24,981 bytes, 86 µs, 125 KiB |

The design this replaces encoded the whole 16 s again for each decision, 77 to 120 ms of
CPU. A body is the audio, the turn and its framing: 476 bytes here, with one line of
conversation, and the turn at most 64 KiB. `ROUTER_CHORUS_ASR_AUDIO=pcm` sends what the
before column does.

**What the model hears.** 20 recordings of requests, acknowledgements and pauses, from 0.5
to 43 s, decided over PCM and over the router's Ogg Opus on the same server: with the floor
free as recorded, with the floor free after one Opus generation at 16 kHz and 32 kb/s, as a
call's audio arrives, and over the agent. The answer was the same in all 60. `p_eot` moved
by a median of 0.011 as recorded and 0.003 after a call's coding, at most 0.14 and 0.05.
Encoding at complexity 5 rather than the default 10 costs 78 µs a frame, and agreed as
well on these 20; at 24 kb/s `p_eot` moved by up to 0.19. The server decodes 16 s of Opus
in 4.9 ms, or 6.3 ms built with `GOEXPERIMENT=simd`.

**The server.** CPU, one lane, weights in f16, each including its request, median of 5:

| | Opus | PCM |
| --- | --- | --- |
| Decision, 3.4 s of audio | 91.7 ms, 9,397 bytes | 90.9 ms, 90,005 bytes |
| Decision, a full 16 s window | 373 ms, 15,568 bytes | 372 ms, 512,405 bytes |

Each answer was the same in both. In a later run on a busier machine, where the 3.4 s
decision took 173 ms, the words alone (`POST /v1/transcribe`, median of 3) took 724 ms
for 3.4 s of audio and 1.12 s for 16.6 s, and the transcriber, on a recorded "Hi, what's
the weather like in Lisbon tomorrow?", wrote "Hi." at the pause after it (880 ms of audio,
in 258 ms), the whole question at its end (2,680 ms, in 655 ms), and ended the utterance
2.5 s later. Audio with no speech in it is answered without the model, in under a
millisecond.

**The transcriber's life.** The same recorded question, then 10 s of silence, in real time
against that server: with digital silence, and with the recording's own room tone, the
transcriber before this change and after it each wrote 2 replacements and 1 final. With 10 s
of line noise at -48 dBFS, the level the old gate took for voice, the old one wrote "Hi." and
never ended the utterance; the new one wrote both replacements and the final.

## Limits

- With another transcriber the pause is timed from the transcript's last change, which
  lags the caller's last sound by the transcriber's own delay, and that transcriber writes
  at its own pace, so a caller who talks on for longer than the cap without a pause it
  writes at is cut off. The policy was fitted to pauses timed from the voice, which
  `stream/chorus-asr` keeps in every mode.
- With another transcriber, a server that answers none of three checks in a row over 8 s
  hands the participant's turns to that transcriber, which ends them as it would with no
  floor. The floor asks the server again 10 s later, then at waits doubling to a minute, and
  takes the turns back at the caller's next quiet moment (500 ms, with nothing the other
  transcriber has yet to settle) once it answers. A check over the agent that runs out of its
  shorter deadline counts neither way. `router_chorus_asr_floor_handovers_total` counts the
  handovers, `to` the transcriber and back to `chorus-asr`.
- The transcriber's speech gate was not measured on calls. Over noise as loud as -48 dBFS
  a pause is heard about 200 ms late, once the detector's score has let go of the voice,
  and a sudden loud noise can score as speech. The gate is absolute: somebody else talking
  in the room 18 dB below the caller keeps it open, so the caller's turn ends at the room's
  pauses, if the model ends it there, or at 60 s, and with no languages declared the model
  writes the room's words into the caller's final. With the agent quiet it writes words
  only at pauses. It rewrites the whole utterance at each pause, ends one at 60 s if the
  caller never pauses, and drops a ninth request waiting on a slow server.
- With no languages declared the words so far are the model's guess on a second or two of
  speech until the utterance's first pause, whose words say what language it is in; the
  language is the model's guess on that pause's words, which the rest of the utterance's
  interims are then held to.
- Checks over the agent cost a decision and the words every `interval_ms` while a caller
  is voiced over it, the agent's own echo included: a line with no echo cancellation costs
  that for as long as the agent talks, on the replay's TPU about three requests a second of
  90 to 140 ms each, each ruled `ignore` and its words dropped as the echo.
  A check's window ends mid-word, where the model's barge-in training clips ended 100 to
  300 ms after the caller's last word, so its ruling there is less sure than at a pause.
- A pause of speech costs a words-bearing request, the agent's own echoed voice included, and the
  words of a long utterance hold a lane for seconds on a CPU.
- Interims are best effort: a busy server refuses them, one is out at a time, and on a CPU
  server the words of a long utterance take seconds, so they come late and seldom there. The
  words the turn is answered on are the final's, whatever the interims showed.
- A check is asked once the one before has answered, so on a server slower than
  `interval_ms` a decision, as a CPU is over a long utterance's 16 s window, checks come late,
  and a late one starts a run of checks in a row again: a point that asks for two then ends
  the turn at `cap_ms`. The replay's 19.8 s turn was answered 3.1 s after the voice under
  `max_wait_ms: 1400` so.
- Echo is dropped by its words, not its sound, and only an utterance's start is timed: the
  model writes no word's time, nor does the voice, and the agent's reckoning of where its
  voice is in the reply, by its speaking rate on the call, is off by up to a third (5th to
  95th percentile over 996 lines of a τ³-Voice bench), too rough to time a word by. An echo
  the model writes too far from what the agent said, or without three of its words heard
  right in a row, is still the caller's, as is one between two or more words of the caller's
  own on each side, even one that truly came back down the line: an echo whose ends the model
  wrote wrong two words at a time, "Tomorrow, and listen, it will be sunny … in the after
  noon.", is taken whole for the caller's, where one word wrong at an end ("Signed and warm
  …") is not. A caller who says three or more of the agent's words in a row while it speaks,
  or in an utterance that began (300 ms before their voice) within 800 ms of its last audible
  sample going out, loses them at the start or the end of what they said so far, or next to
  a murmur or a lone word: "Okay, the tablet order, please." over a reply that says "the
  tablet order" is "Okay, please.", and "Just use the address from the tablet order." loses
  all but "Just use" until "…for every—" is written. A word the model writes with some of
  the agent's in it goes whole: "No, it's A-J-A-M-E-L." over a reply that spelled
  "J-A-M-E-L" is "No, it's". One word of the caller's between two runs of the echo reads as
  a word of the agent's misheard. Two or more words in a row are the caller's wherever they
  fall, as "and in Porto" does inside the reply's echo.
- With unified replies the reply rules on its own words and no flow controller is asked,
  so acting has nothing to stand in for.
- Without turns from the audio, a CPU decision over a full window, 373 ms here, does not fit
  the 250 ms an overlap gives it by default, so words over the agent fall back to the flow
  controller on such a server. Acting with chorus-asr transcribing, a pause over the agent is
  settled words, with 1 s.

## Test it

```bash
ROUTER_CHORUS_ASR_URL=http://127.0.0.1:8080 CHORUS_ASR_WAV=caller.wav \
  go test -tags integration -run TestARealServer -v ./internal/audiofloor/
```

The acceptance replay plays recorded calls into a real agent in real time
(`internal/agent/audio_floor_integration_test.go`), with chorus-asr transcribing and
acting, and fails on what a caller would hear go wrong; each call's timeline is in the
output and its debug log in `CHORUS_ASR_REPLAY_LOGS`:

```bash
ROUTER_CHORUS_ASR_URL=http://127.0.0.1:8080 GOEXPERIMENT=simd \
  go test -tags integration -count=1 -timeout 30m -run TestAudioFloorReplay -v ./internal/agent/
```

Both read `ROUTER_CHORUS_ASR_API_KEY` too, for a server behind a gateway; the replay counts
its requests by route under the gateway's path. The router needs only the URL: each call's
agent chooses its own floor control, as agents on one router do, acting unless a check says
otherwise (transcribing alone with the default flow controller).

`TestAudioFloorScenarios`, in `audio_floor_scenarios_integration_test.go`, plays what the
acceptance checks don't: pauses inside a turn, a correction, a question and a second turn
over the reply, silence after the agent asks a question, a thinking murmur inside a long
pause, a cough over the agent, a caller who goes on after being answered, the server slow or
answering 503, the agent speaking first, a monologue of over 60 s and a caller who switches
language. Each is a call of its own, run by its number (`-run 'TestAudioFloorScenarios/S4'`),
and logs a `row:` line of what the loop did and how long it took; what each one measured on
the TPU server is in [Scenarios](#scenarios). The speech is rendered by
`testdata/audiofloor/render.sh`.
