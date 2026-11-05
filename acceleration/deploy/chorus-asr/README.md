# chorus-asr

One speech model that hears the caller's last 16 s and answers the flow controller's
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
- **Both, acting.** An agent that acts on the model and names it as its STT has the
  transcriber time the caller's pauses from the audio and ask about each, the decision and
  the words in one request: [Turns from the audio](#turns-from-the-audio).

## Configure

| Variable | Purpose |
| --- | --- |
| `ROUTER_CHORUS_ASR_URL` | The server, such as `http://floor:8080`. Adds the `stream/chorus-asr` transcriber |
| `ROUTER_CHORUS_ASR_MODE` | `off` (default); `shadow`, which asks the model about each turn beside the flow controller and acts on nothing it says; or `act`, which asks it first and acts on its ruling |
| `ROUTER_CHORUS_ASR_AUDIO` | `opus` (default), or `pcm`, 16-bit PCM at 8 to 20 times the bytes, to tell the codec apart from the model when debugging a decision |

`cmd/router` and `cmd/agent` read these the same way (`audiofloor.Open`).

The server is started with its model and the `policy.json` that ships with that model.
`GET /v1/info` hands the policy to the router, so nothing about a checkpoint is in the
router. A server without one answers 501, and the router never acts on it.

## Wire protocol

`GET /v1/info`: what the server serves, here with each preset's `measured` left out:

```
{"template":"v3.5","conversation_on_free_floor":false,"end_of_turn":{"first_check_ms":200,"first_pause_logit":0,"interval_ms":300,"cap_ms":2500,"default":"safe","presets":{"fast":{"bars":[{"from_ms":0,"bar":0.8},{"from_ms":500,"bar":0.3}],"measured":{...}},"safe":{"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"measured":{...}}}},"max_tools":21,"sample_rate":16000,"window_samples":256000}
```

`end_of_turn` is fitted to the server's model. A pause is checked `first_check_ms` into it,
counted from the voice's last frame, and every `interval_ms` after, when the transcriber is
chorus-asr too ([Turns from the audio](#turns-from-the-audio)); with another transcriber the
first check is when the words have held still for the cadence's gap, 350 ms after they last
changed, and the rest every `interval_ms` after. The turn has ended at the first check where `p_eot`
reaches the bar of the last of a preset's `bars` to have begun by then, `from_ms` into the
pause, raised by `first_pause_logit` in log-odds on the turn's first pause. At `cap_ms` of
pause it has ended whatever the model says. The presets are how eager turn-taking is, and
the router applies the one the server names as its `default`. These values belong to one
checkpoint, and another one comes with its own. The presets of the current one, on held-out
calls:

| Preset | Bars | Cut-ins | Answers at (median, p90) | Real dictation pauses cut | Slow counts |
| --- | --- | --- | --- | --- | --- |
| `safe` (default) | 0.995 from 0 ms, 0.20 from 1,100 ms | 4.5% | 1,100 ms, 1,100 ms | 12.6% | none of 9 cut |
| `fast` | 0.8 from 0 ms, 0.3 from 500 ms | 14.5% | 500 ms, 800 ms | within the limit | one cut after "One" |

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
so a server started without a policy transcribes as well, and so it does in `shadow`.

## Turns from the audio

Acting (`ROUTER_CHORUS_ASR_MODE=act`) with `stream/chorus-asr` as the agent's STT, the turn
ends when the model says so, timed from the voice, and a pause costs one request:

- **Pauses from the audio.** The transcriber's speech gate (-48 dBFS and the speech
  detector, below) says when the voice stopped. The transcriber asks about the pause
  `first_check_ms` (200 ms) after the voice's last frame, then every `interval_ms`, each
  once the one before has answered, on the agent's behalf: the moment of the call is the
  one the flow controller would put (`audiofloor.Asker`, which the agent hands its
  transcriber when it opens it).
- **One request a pause.** The first check of a pause is `POST /v1/decide?agent=ID&transcript=true`
  with the whole utterance: the server decides on its last 16 s, answers that at once, and
  writes the words of all of it in the same pass. The rechecks ask for the decision alone,
  on what a decision hears. The agent is registered with the session's languages, for the
  words.
- **One final.** A check that ends the turn hands its ruling to the agent, and its words
  go out as the utterance's one final, to the cadence as any final does (60 ms). The flow
  controller's Hearer rules on those words as the check did, with no request of its own, and
  the reply is asked for at once. Over the agent a check has no bar: the model's ruling ends
  the turn, `stop`, `shorten`, `continue` or `ignore`, and words heard over a reply reach the
  flow controller as settled words.
- **Nothing in between.** Words of a pause the model waits on are kept for its end, and
  nothing is written while the caller talks, so nothing supersedes a ruling or interrupts a
  reply about audio already ruled on. Speech after a pause the model waited on begins
  another: its first check writes the whole utterance again.

Any other combination is as it was: other transcribers, `shadow`, and transcription
alone. A check that fails, or no check ending the pause within 1 s of the transcriber's own
end at 2.5 s, leaves that pause to the words alone and the flow controller, as before. A
check that ended the turn is waited on for its words however long they take, which on a
CPU is seconds for a long turn.

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
ended; with the floor free the router has it ended when `p_eot` reaches the bar for this
check, and otherwise rules wait and puts the same words again after `interval_ms`. The
first check of a pause is the first time its words are put (`FlowTurn.Waited` false), the
turn's first pause is one before any wait, and the pause runs from the voice's last frame
when the transcriber is chorus-asr too, and otherwise from when the words last changed.

| Model | `Decided.Disposition` | `Decided.Floor` | Also |
| --- | --- | --- | --- |
| floor free, `p_eot` under the preset's bar and the pause under the cap | `wait` | `continue` | `Again`: `interval_ms` |
| `respond`, `clarify`, `ignore` | as said | `continue` | |
| `cannot`, or a tool | `respond` | `continue` | |
| over the agent: `stop`, `shorten` | `wait` while unfinished, else `respond` | as said | |
| over the agent: `continue`, `ignore` | `ignore` | `continue` | |

The tool the caller wants, and `cannot`, are in the shadow log; nothing acts on them, and a
tool's approval is untouched. Acting, a decision's `Ruler` is `heard`.

## Fallbacks

- **Acting.** The model has a third of the flow controller's deadline: 1 s for a pause,
  250 ms for words over the agent. A timeout, an error, a refused agent (more than 21
  customer tools), or a server without a policy leaves the turn to the flow controller,
  language model or judge, with the other two thirds. `router_chorus_asr_checks_total`
  counts each check by outcome; acting, every outcome but `answered` and `superseded` is a
  turn the flow controller decided.
- **Shadow.** One question at a time, none waited for: a slow server costs a call nothing.
  `router_agent_chorus_asr_compared_total` counts both rulings by disposition and floor,
  and the preset, and a debug log line carries `p_end`, the preset and its bar, the pause,
  first check and first pause.
- **Transcripts.** A server that is busy, restarting or slow costs a request, not the
  session: a 5xx, 404, 408 or 429, a timeout or a reset connection is logged and counted in
  `router_chorus_asr_transcribe_failures_total` by status, `timeout` or `connection`, and
  the utterance's words are asked for again, whole, at its end or its next pause after more
  speech. Only a server the router is misconfigured for (another 4xx, or no policy where
  one is needed) is fatal, which hands the call to the next transcriber the target names;
  a target that names this transcriber alone has no next one. A request for words times
  out at 8 s and the audio's length again.
- **Echo.** A phone with no echo cancellation sends the agent's voice back down the
  caller's line, and the model writes it as the caller's words, or adds it to theirs within
  one utterance. The agent drops a chorus-asr transcript's end that is three or more of its
  own words, near enough (a quarter of them may be heard wrong), when the audio overlapped
  its speech or came within 1.5 s of it: what is left is the caller's, and an echo alone is
  nothing heard (`router_agent_chorus_asr_echoes_total`). It does so in every mode; the
  transcriber itself still writes what it heard. Other transcribers are left as they are.
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

**The acceptance replay** (`TestAudioFloorReplay`), against one CPU server, f16, 2 lanes,
on a machine shared with other work: macOS `say` speech at 16 kHz in real time through the
agent, the voice a recorded 8.8 s reply, and the flow controller's language model answering
wrong on purpose, so that a check passes only on the model's rulings. Before is the
transcriber writing at pauses and the Hearer asked 350 ms after the words held still; after
is turns from the audio.

| Check | Before (`safe`) | After, `safe` | After, `fast` |
| --- | --- | --- | --- |
| 1. "Uh-huh" over the reply | reply stopped: the decision took 251 ms of the 250 over the agent, and the fallback ruled | reply whole, no new answer | the same |
| 2. One final after 10 s of silence, room tone, line noise at -50 dBFS | pass | pass, one request for the words | pass |
| 2. The reply echoed at -20 dB | two more answers, to its own echo, and three replies interrupted | the conversation as with no echo; the transcriber still writes the echo | the same |
| 2. Echo at -30 dB acting, -20 dB in `shadow` and transcribing alone | not run | the conversation as with no echo | the same |
| 3. A breath or a burst of noise after a turn | no other script | no other script | no other script |
| 4. One to ten, a second apart | answered "One." (p_eot 0.9986 against 0.995) | answered after "ten" | answered "One." (0.857 against 0.8), as `fast` is measured to |
| 5. 19.8 s with pauses under a second | answered four times mid-turn: the words held still while the caller talked on, and the pause reached the cap | one final with its beginning, one answer, 3.4 s after the voice | the same, 3.1 s |
| 6. A cough alone | nothing | nothing | nothing |
| 7. A new agent registers | 265 ms, inside the 1 s | 388 ms | 248 ms |
| 9. Transcribing alone | pass | pass | pass |

From the end of the caller's voice, "Will it rain in Porto this weekend?" (four calls) and
"Hi, what's the weather like in Lisbon tomorrow?" (two):

| | Before | After, `safe` | After, `fast` |
| --- | --- | --- | --- |
| The server's ruling | | 309 to 328 ms; 539 and 718 ms | 306 to 325 ms; 433 and 584 ms |
| The turn answered (ruling and words) | 1.08 to 1.10 s; 1.39 and 1.45 s | 664 to 771 ms; 1.03 and 1.20 s | 656 to 804 ms; 0.95 and 1.16 s |
| Requests a pause | 2: the words, then the decision | 1 | 1 |

The weather question pauses after "Hi", which is asked about first, and the second call is
a new agent's first turn, registered then. Both questions end with p_eot over 0.995 at the
first check, so the two presets answer them alike. A breath after a question restarts the
pause, and a turn is answered 200 ms after the breath, about 1.1 s after the words.

## Limits

- Without [Turns from the audio](#turns-from-the-audio) the pause is timed from the
  transcript's last change, which lags the caller's last sound by the transcriber's own
  delay, and the transcriber writes only at pauses, so a caller who talks on for longer than
  the cap without a 200 ms pause is cut off. The policy was fitted to pauses timed from the
  voice.
- The transcriber's speech gate was not measured on calls. Over noise as loud as -48 dBFS
  a pause is heard about 200 ms late, once the detector's score has let go of the voice,
  and a sudden loud noise can score as speech. It writes words only at pauses, so words said over the agent reach
  the flow controller at the caller's next pause. It rewrites the whole utterance at each
  pause, ends one at 60 s if the caller never pauses, and drops a ninth request waiting on
  a slow server.
- With the transcriber and the Hearer both on but not acting, a pause costs two requests:
  the words, and the decision. Acting, every pause of speech costs a words-bearing request,
  the agent's own echoed voice included, and the words of a long utterance hold a lane for
  seconds on a CPU.
- Acting with chorus-asr as the transcriber, nothing is written while the caller talks: the
  words appear when the turn ends.
- Echo is dropped by its words, not its sound: an echo the model writes too far from what
  the agent said, or of fewer than three words, is still the caller's; and a caller who
  repeats three or more of the agent's words while it speaks, or within 1.5 s, loses them.
- With unified replies the reply rules on its own words and no flow controller is asked,
  so acting has nothing to stand in for; shadow compares the model with the reply's ruling.
- Without turns from the audio, a CPU decision over a full window, 373 ms here, does not fit
  the 250 ms an overlap gives it, so words over the agent fall back to the flow controller
  on such a server. Acting with chorus-asr transcribing, a pause over the agent is settled
  words, with 1 s.

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
