# An audio floor model

One speech model that hears the caller's last 16 s and answers the flow controller's
question from the audio: whether the caller has finished, what to do with the words
(respond, clarify, ignore), what to do about the agent's own speech (stop, shorten,
continue), and which of the agent's tools is wanted. The same server can write down what
was said. The router speaks to it as `internal/audiofloor`.

It plugs in twice, and each is optional:

- **Decisions.** The flow controller's `Hearer` (`internal/harness/hearer.go`), beside
  `Judge`. A `Judge` is asked named questions about written state; this model is asked
  about the audio, in the words it was trained on, which only its server can write.
- **Transcripts.** The `audiofloor/audio-floor` transcriber, an `stt.STT` like any other,
  chosen by an agent that names it as its STT. With it a call runs no other transcriber.
  At each pause of `first_check_ms` it writes the whole utterance so far, as a replacement
  of the words before, and it ends the utterance at a pause of `cap_ms`. When to answer
  stays the flow controller's.

## Configure

| Variable | Purpose |
| --- | --- |
| `ROUTER_AUDIO_FLOOR_URL` | The server, such as `http://floor:8080`. Adds the `audiofloor/audio-floor` transcriber |
| `ROUTER_AUDIO_FLOOR_MODE` | `off` (default); `shadow`, which asks the model about each turn beside the flow controller and acts on nothing it says; or `act`, which asks it first and acts on its ruling |
| `ROUTER_AUDIO_FLOOR_AUDIO` | `opus` (default), or `pcm`, 16-bit PCM at 8 to 20 times the bytes, to tell the codec apart from the model when debugging a decision |

`cmd/router` and `cmd/agent` read these the same way (`audiofloor.Open`).

The server is started with its model and the `policy.json` that ships with that model.
`GET /v1/info` hands the policy to the router, so nothing about a checkpoint is in the
router. A server without one answers 501, and the router never acts on it.

## Wire protocol

`GET /v1/info`: what the server serves, here with each preset's `measured` left out:

```
{"template":"v3.5","conversation_on_free_floor":false,"end_of_turn":{"first_check_ms":200,"first_pause_logit":0,"interval_ms":300,"cap_ms":2500,"default":"safe","presets":{"fast":{"bars":[{"from_ms":0,"bar":0.8},{"from_ms":500,"bar":0.3}],"measured":{...}},"safe":{"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"measured":{...}}}},"max_tools":21,"sample_rate":16000,"window_samples":256000}
```

`end_of_turn` is fitted to the server's model. A pause is checked `first_check_ms` into it
and every `interval_ms` after, and the turn has ended at the first check where `p_eot`
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
answers. The transcriber's agent is told nothing and names the session's language hints
as `languages`, the languages its callers speak. The ID follows the content, so an agent
registers once and again only when it changes or the server restarts (a 404).

```
{"instructions":"You answer the phone for a weather service.","tools":[{"name":"get_weather","description":"Get the forecast for a city.","parameters":{"type":"object","required":["city"]}}]}
{"agent":"eb4fb3f2cb56bb977b7f0376680a4aa5","template":"v3.5","options":["respond","wait","clarify","ignore","cannot","get_weather"],"warmed_lanes":1,"warm_ms":67.85}
```

`POST /v1/decide?agent=ID`: a `multipart/form-data` body of a `turn` part, the moment of
the call as UTF-8 JSON, then an `audio` part: Ogg Opus (`audio/ogg`), or with
`ROUTER_AUDIO_FLOOR_AUDIO=pcm` 16 kHz mono PCM (`audio/pcm;rate=16000;channels=1;format=s16le`).
`transcript=true` adds a second line, whose text is empty when nothing was said.

```
turn: {"conversation":[{"agent":true,"text":"Weather line, how can I help?"}]}

{"answer":"get_weather","answer_probability":0.9988813,"p_eot":0.9999969,"options":[{"name":"respond","p":0.00015754762},{"name":"wait","p":0.0000031085704},{"name":"clarify","p":0.00080397225},{"name":"ignore","p":0.0000036620925},{"name":"cannot","p":0.00015045748},{"name":"get_weather","p":0.9988813}],"sample_rate":16000,"samples":35364,"window_samples":35364,"decide_ms":102.45,"ruling":{"disposition":"respond","floor":"continue","tool":"get_weather"}}
{"transcript":{"text":"Hi, what's the weather like in Lisbon tomorrow?","language":"English"},"transcript_ms":311.4}
```

The turn's fields are `speaking`, `said` (the agent's reply so far), `unfinished`,
`speaker` (`caller` or `another_voice`; a participant's name never leaves the router) and
`conversation`. The router sends what a prompt shows: the reply's last 800 characters, the
lines of the last six turns, and each line's last 40 words. The turn goes in the body
because the longest, 800 characters of reply and six lines of 40 words, escapes to 10,285
bytes in Hindi and 12,035 in Russian as JSON in a header, past the 8 KiB a proxy allows one
header line.

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
turn's first pause is one before any wait, and the pause runs from when the words last
changed.

| Model | `Decided.Disposition` | `Decided.Floor` | Also |
| --- | --- | --- | --- |
| floor free, `p_eot` under the preset's bar and the pause under the cap | `wait` | `continue` | `Again`: `interval_ms` |
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
  language model or judge, with the other two thirds. `router_audio_floor_checks_total`
  counts each check by outcome; acting, every outcome but `answered` and `superseded` is a
  turn the flow controller decided.
- **Shadow.** One question at a time, none waited for: a slow server costs a call nothing.
  `router_agent_audio_floor_compared_total` counts both rulings by disposition and floor,
  and the preset, and a debug log line carries `p_end`, the preset and its bar, the pause,
  first check and first pause.
- **Transcripts.** A server that is down when the session starts, or that fails a
  request, is a fatal error, which hands the call to the next transcriber the target names.
  The words since the last ones written are lost. A target that names this transcriber
  alone has no next one.
- **Noise.** The transcriber drops nothing itself. The server returns empty text for audio
  with no speech in it, for words in a language the callers don't speak, and for a few words
  of audio its decision ruled `ignore`, which is how a cough or a breath comes out. Empty
  text is nothing heard: it writes nothing, and an utterance ends on the last words that
  weren't empty. The transcriber's level gate only finds the pauses.
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

**The server.** CPU, one lane, weights in f16, each including its request, 5 decisions and
3 transcripts:

| | Opus | PCM |
| --- | --- | --- |
| Decision, 3.4 s of audio | 91.7 ms, 9,397 bytes | 90.9 ms, 90,005 bytes |
| Decision, a full 16 s window | 373 ms, 15,568 bytes | 372 ms, 512,405 bytes |
| Decision and transcript, 3.4 s | 353 ms | |
| Decision and transcript, 16.6 s | 848 ms | |

Each answer was the same in both. The transcriber, on a recorded "Hi, what's the weather
like in Lisbon tomorrow?", wrote "Hi." at the pause after it (860 ms of audio, in 193 ms),
the whole question at its end (2,680 ms, in 329 ms), and ended the utterance 2.5 s later.

## Limits

- The pause is timed from the transcript's last change, which lags the caller's last sound
  by the transcriber's own delay. The policy was fitted to pauses timed from the voice.
- The transcriber's level gate, 0.004 of full scale, which finds its pauses, was not
  measured on calls. It writes words only at pauses, so words said over the agent reach
  the flow controller at the caller's next pause. It rewrites the whole utterance at each
  pause, ends one at 60 s if the caller never pauses, and drops a ninth request waiting on
  a slow server.
- With both modes on, a pause costs two requests: the transcript asked for the default
  agent, and the decision asked for the agent's own.
- With unified replies the reply rules on its own words and no flow controller is asked,
  so acting has nothing to stand in for; shadow compares the model with the reply's ruling.
- A CPU decision over a full window, 373 ms here, does not fit the 250 ms an overlap
  gives it, so words over the agent fall back to the flow controller on such a server.

## Test it

```bash
ROUTER_AUDIO_FLOOR_URL=http://127.0.0.1:8080 AUDIO_FLOOR_WAV=caller.wav \
  go test -tags integration -run TestARealServer -v ./internal/audiofloor/
```
