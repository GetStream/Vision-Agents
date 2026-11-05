# chorus-asr

chorus-asr is one speech model that hears the caller's last 16 s and answers the flow controller's
question from the audio: whether the caller has finished, what to do with the words
(respond, clarify, ignore), what to do about the agent's own speech (stop, shorten,
continue), and which of the agent's tools is wanted. The same server can write down what
was said. The router speaks to it as `internal/audiofloor`.

It plugs in twice, and each is optional:

- **Decisions.** The flow controller's `Hearer` (`internal/harness/hearer.go`), beside
  `Judge`. A `Judge` is asked named questions about written state; this model is asked
  about the audio, in the words it was trained on, which only its server can write.
- **Transcripts.** The `stream/chorus-asr` transcriber, an `stt.STT` like any other,
  chosen by an agent that names it as its STT. With it a call runs no other transcriber.
  At each pause of 200 ms it writes the whole utterance so far, as a replacement of the
  words before, and it ends the utterance at a pause of 2.5 s. When to answer
  stays the flow controller's.

## Configure

| Variable | Purpose |
| --- | --- |
| `ROUTER_CHORUS_ASR_URL` | The server, such as `http://floor:8080`. Adds the `stream/chorus-asr` transcriber |
| `ROUTER_CHORUS_ASR_MODE` | `off` (default); `shadow`, which asks the model about each turn beside the flow controller and acts on nothing it says; or `act`, which asks it first and acts on its ruling |
| `ROUTER_AUDIO_FLOOR_AUDIO` | `opus` (default), or `pcm`, 16-bit PCM at 8 to 20 times the bytes, to tell the codec apart from the model when debugging a decision |

`cmd/router` and `cmd/agent` read these the same way (`audiofloor.Open`).

The server is started with its model and the `policy.json` that ships with that model.
`GET /v1/info` hands the policy to the router, so nothing about a checkpoint is in the
router. A server without one answers 501, and the router never acts on it.

Three ways to use it, each a short Go program in
[examples/routers/chorus_asr_example](../../../examples/routers/chorus_asr_example/README.md):

| Use | Router | Agent |
| --- | --- | --- |
| [Decisions and transcripts](../../../examples/routers/chorus_asr_example/decide_and_transcribe/main.go): one model hears the caller, writes the words and ends the turn | `ROUTER_CHORUS_ASR_URL`, `ROUTER_CHORUS_ASR_MODE=act` | `stt: stream/chorus-asr`, its own `tts` |
| [Decisions only](../../../examples/routers/chorus_asr_example/decide_only/main.go): another transcriber writes the words | `ROUTER_CHORUS_ASR_URL`, `ROUTER_CHORUS_ASR_MODE=act` | any `stt` |
| [Transcription only](../../../examples/routers/chorus_asr_example/transcribe_only/main.go): the words, through the router's STT | `ROUTER_CHORUS_ASR_URL` | a router config naming `stream/chorus-asr` |

## How eager turn-taking is: `end_of_turn`

How long chorus-asr waits before it takes a pause for the end of a turn is the app's to
choose, per agent, beside its `stt`, `llm` and `tts`. Waiting longer cuts the caller off less
often, and answers later. `end_of_turn` takes one of three:

```yaml
# agent.yaml
end_of_turn:
  max_wait_ms: 1100   # the measured point with the fewest cut-ins that answers within 1.1 s at p90
```

```yaml
end_of_turn:
  preset: fast        # one of the server's presets, each naming a measured point
```

```yaml
end_of_turn:
  bars:               # your own: the first from 0 ms, each later one further into the pause
    - {from_ms: 0, bar: 0.97}
    - {from_ms: 900, bar: 0.25}
```

Left out, an agent keeps what is stored, and a new one ends its turns by the server's
`default` preset; `end_of_turn: {}` is the default again. A sync (`POST /v1/agents/sync`, as
every SDK's folder sync sends it), a config's create, replace and patch, and a server-side
`createSession`, which chooses for one session over its config's, all take the same field.

The server measured a frontier of points on held-out calls, and these are the points of the
checkpoint this was written against. Each point's bars apply from `from_ms` into a pause, and
the turn has ended once as many checks of the pause in a row as its "In a row" have reached
their bar:

| Point | Chosen by | Bars | In a row | Cut-ins | Answers at (median, p90) | Real dictation pauses cut | Slow counts cut | Dictation guard |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| 500 ms | `preset: fast`, `max_wait_ms` 800 to 1,099 | 0.8 from 0 ms, 0.3 from 500 ms | 1 | 14.5% | 500 ms, 800 ms | 16.0% | 9 of 9 | fails |
| 1,100 ms | `preset: safe` (default), `max_wait_ms` 1,100 to 1,399 | 0.995 from 0 ms, 0.2 from 1,100 ms | 1 | 4.5% | 1,100 ms, 1,100 ms | 12.6% | 0 of 9 | passes |
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
- **What is checked, and when.** A write is refused (400), before anything is stored, when
  `end_of_turn` names more than one of the three, when its bars don't begin at 0 ms and rise,
  or one isn't strictly between 0 and 1, when the server publishes no preset by that name,
  or when `max_wait_ms` is shorter than every measured point's p90 wait. A device may not
  choose a session's (403). A sync that chooses a point failing the dictation guard is stored,
  and its `warnings` say what the point does to slow counts and dictation pauses. When the
  router asks chorus-asr nothing about turns (`ROUTER_CHORUS_ASR_MODE` off), or its server
  can't say what it publishes, the choice is kept and the sync's `warnings` say so.
- **What it replaces.** A point brings its bars and its checks in a row; bars of your own end a
  turn at the first check that reaches its bar. When a pause is checked, how much more a turn's
  first pause asks, and the cap stay the server's, which are fitted to its model. A run of checks
  is one pause's, one after another: the caller speaking, a check not asked about or not
  answered, or one under its bar starts it again.
- **When it is read.** At each check, against the policy of the model the server runs then.
  A preset or wait that model no longer publishes, as after a new checkpoint, ends the
  session's turns by its default, and a warning says so once per session.
- **What you see.** Once a session, and again if it changes, the router logs the point its
  turns end by: `end_of_turn`, `checks_in_a_row`, `dictation_guard`, and the point's
  `max_wait_ms`, `cut_ins`, `answer_ms_p90`, `dictation_pauses_cut`, `slow_counts_cut` and
  `slow_counts`. `router_agent_chorus_asr_compared_total` counts each shadow comparison by its
  `end_of_turn` label, `preset:NAME`, `max_wait_ms:N` for the point fitted to `N` (whatever
  wait the agent asked for) or `custom` for its own, and its `dictation_guard` label, `pass`,
  `fail` or `unmeasured`; the shadow log line carries both, with the bar it was held to and
  the checks in a row so far.

Bars of your own were not measured: measure them on your own calls before relying on them.

## Wire protocol

`GET /v1/info`: what the server serves, here with two of its three points left out:

```
{"template":"v3.5","conversation_on_free_floor":false,"end_of_turn":{"first_check_ms":200,"interval_ms":300,"cap_ms":2500,"first_pause_logit":0,"fitted_to":"...","measured_on":"...","default":"safe","presets":{"safe":1100,"fast":500},"frontier":[...,{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":1,"measured":{"cut_ins":0.0454,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.1257,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}},...]},"max_tools":21,"sample_rate":16000,"window_samples":256000}
```

`end_of_turn` is fitted to the server's model. A pause is checked `first_check_ms` into it
and every `interval_ms` after. At each check the bar is that of the last of the point's
`bars` to have begun, `from_ms` into the pause, raised by `first_pause_logit` in log-odds on
the turn's first pause, and the check reaches it when `p_eot` is at least the bar. The turn
has ended at the first check that reaches its bar with the `checks_in_a_row - 1` checks of
the pause before it, and at `cap_ms` of pause whatever the model says. `frontier` is the
measured points, `presets` name points by their `max_wait_ms`, and `default` is the preset
for an agent that chooses none (see
[How eager turn-taking is](#how-eager-turn-taking-is-end_of_turn)). A point's `measured` is
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
`ROUTER_AUDIO_FLOOR_AUDIO=pcm` 16 kHz mono PCM (`audio/pcm;rate=16000;channels=1;format=s16le`).

```
turn: {"conversation":[{"agent":true,"text":"Weather line, how can I help?"}]}

{"answer":"get_weather","answer_probability":0.9988813,"p_eot":0.9999969,"options":[{"name":"respond","p":0.00015754762},{"name":"wait","p":0.0000031085704},{"name":"clarify","p":0.00080397225},{"name":"ignore","p":0.0000036620925},{"name":"cannot","p":0.00015045748},{"name":"get_weather","p":0.9988813}],"sample_rate":16000,"samples":35364,"window_samples":35364,"decide_ms":102.45,"ruling":{"disposition":"respond","floor":"continue","tool":"get_weather"}}
```

`POST /v1/transcribe?languages=en`: the audio as the body, as a decision's audio part, and
the languages the callers speak, the session's language hints. The answer is what was
said, of all of the audio, with no agent, decision or policy:

```
{"transcript":{"text":"Hi, what's the weather like in Lisbon tomorrow?","language":"English"},"transcript_ms":311.4}
```

The turn's fields are `speaking`, `said` (the agent's reply so far), `unfinished`,
`speaker` (`caller` or `another_voice`; a participant's name never leaves the router) and
`conversation`. The router sends what a prompt shows: the reply's last 800 characters, the
lines of the last six turns, and each line's last 40 words. The turn goes in the body
because the longest, 800 characters of reply and six lines of 40 words, escapes to 10,285
bytes in Hindi and 12,035 in Russian as JSON in a header, past the 8 KiB a proxy allows one
header line.

## Transcription alone

The transcriber needs only `ROUTER_CHORUS_ASR_URL`. With `ROUTER_CHORUS_ASR_MODE=off`,
an agent that names `stream/chorus-asr` as its `stt` gets the model's words and the
flow controller it would have had anyway:

```yaml
stt: stream/chorus-asr
```

It registers no agent and asks for no decision: each request is `POST /v1/transcribe`,
so a server started without a policy transcribes as well. With the mode on, the Hearer
still asks for its decisions on its own, and the transcriber still asks for the words
alone: keeping the words a decision could carry would tie the transcriber's pauses to the
flow controller's turns, for one request saved a pause.

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
turn's first pause is one before any wait, and the pause runs from when the words last
changed.

| Model | `Decided.Disposition` | `Decided.Floor` | Also |
| --- | --- | --- | --- |
| floor free, fewer checks in a row at the agent's bar than its point asks, and the pause under the cap | `wait` | `continue` | `Again`: `interval_ms` |
| `respond`, `clarify`, `ignore` | as said | `continue` | |
| `cannot` | `respond` | `continue` | `Cannot` |
| a tool | `respond` | `continue` | `Tool` |
| over the agent: `stop`, `shorten` | `wait` while unfinished, else `respond` | as said | |
| over the agent: `continue`, `ignore` | `ignore` | `continue` | |

`Tool` and `Cannot` are hints on the decision. Nothing runs a tool from them, and a tool's
approval is untouched. Acting, a decision's `Ruler` is `heard`.

## Fallbacks

- **Acting.** The model has a third of the flow controller's deadline: 1 s for a pause,
  250 ms for words over the agent. A timeout, an error, a refused agent (more than 21
  customer tools), or a server without a policy leaves the turn to the flow controller,
  language model or judge, with the other two thirds. `router_chorus_asr_checks_total`
  counts each check by outcome; acting, every outcome but `answered` and `superseded` is a
  turn the flow controller decided.
- **Shadow.** One question at a time, none waited for: a slow server costs a call nothing.
  `router_agent_chorus_asr_compared_total` counts both rulings by disposition and floor,
  and the `end_of_turn` and `dictation_guard` it was read by, and a debug log line carries
  `p_end`, the same two, the bar and the checks in a row, the pause, first check and first
  pause.
- **Transcripts.** A server that is down when the session starts, or that fails a
  request, is a fatal error, which hands the call to the next transcriber the target names.
  The words since the last ones written are lost. A target that names this transcriber
  alone has no next one.
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
- **A restart.** A 404 has the router register again and read `/v1/info` again, once.

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
conversation, and the turn at most 64 KiB. `ROUTER_AUDIO_FLOOR_AUDIO=pcm` sends what the
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

- The pause is timed from the transcript's last change, which lags the caller's last sound
  by the transcriber's own delay. The policy was fitted to pauses timed from the voice.
- The transcriber's speech gate was not measured on calls. Over noise as loud as -48 dBFS
  a pause is heard about 200 ms late, once the detector's score has let go of the voice,
  and a sudden loud noise can score as speech. It writes words only at pauses, so words said over the agent reach
  the flow controller at the caller's next pause. It rewrites the whole utterance at each
  pause, ends one at 60 s if the caller never pauses, and drops a ninth request waiting on
  a slow server.
- With the transcriber and the Hearer both on, a pause costs two requests: the words, and
  the decision.
- With unified replies the reply rules on its own words and no flow controller is asked,
  so acting has nothing to stand in for; shadow compares the model with the reply's ruling.
- A CPU decision over a full window, 373 ms here, does not fit the 250 ms an overlap
  gives it, so words over the agent fall back to the flow controller on such a server.

## Test it

```bash
ROUTER_CHORUS_ASR_URL=http://127.0.0.1:8080 AUDIO_FLOOR_WAV=caller.wav \
  go test -tags integration -run TestARealServer -v ./internal/audiofloor/
```
