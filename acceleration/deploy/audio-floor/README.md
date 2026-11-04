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

The server is started with its model and the `policy.json` that ships with that model.
`GET /v1/info` hands the policy to the router, so nothing about a checkpoint is in the
router. A server without one answers 501, and the router never acts on it.

## Wire protocol

`GET /v1/info`: what the server serves. This example was served with a test policy fitted
to an earlier checkpoint:

```
{"template":"v3.5","conversation_on_free_floor":false,"end_of_turn":{"first_check_ms":200,"first_bar":0.7,"later_bar":0.2,"first_pause_logit":0.5,"interval_ms":300,"cap_ms":2500},"max_tools":21,"sample_rate":16000,"window_samples":256000}
```

`end_of_turn` is fitted to the server's model. A pause is checked `first_check_ms` into it
and every `interval_ms` after, and the turn has ended at the first check where `p_eot`
reaches the bar: `first_bar` at the pause's first check and `later_bar` after, each raised
by `first_pause_logit` in log-odds on the turn's first pause. At `cap_ms` of pause it has
ended whatever the model says. These values belong to one checkpoint, and another one
comes with its own.

`POST /v1/agents`: an agent, by what it was told and its customer tools. The router's own
tools (ending or transferring the call, pressing keys, the knowledge and web lookups,
confirming a tool call, video frames) are left out: listing them changes the model's
answers. The ID follows
the content, so an agent registers once and again only when it changes or the server
restarts (a 404).

```
{"instructions":"You answer the phone for a weather service.","tools":[{"name":"get_weather","description":"Get the forecast for a city.","parameters":{"type":"object","required":["city"]}}]}
{"agent":"eb4fb3f2cb56bb977b7f0376680a4aa5","template":"v3.5","options":["respond","wait","clarify","ignore","cannot","get_weather"],"warmed_lanes":1,"warm_ms":67.85}
```

`POST /v1/decide?agent=ID`: a `multipart/form-data` body of a `turn` part, the moment of
the call as UTF-8 JSON, then an `audio` part, 16 kHz mono PCM
(`audio/pcm;rate=16000;channels=1;format=s16le`). `transcript=true` adds a second line.

```
turn: {"conversation":[{"agent":true,"text":"Weather line, how can I help?"}]}

{"answer":"get_weather","answer_probability":0.9988813,"p_eot":0.9999969,"options":[{"name":"respond","p":0.00015754762},{"name":"wait","p":0.0000031085704},{"name":"clarify","p":0.00080397225},{"name":"ignore","p":0.0000036620925},{"name":"cannot","p":0.00015045748},{"name":"get_weather","p":0.9988813}],"sample_rate":16000,"samples":35364,"window_samples":35364,"decide_ms":102.45,"ruling":{"disposition":"respond","floor":"continue","tool":"get_weather"}}
{"transcript":{"text":"Hi, what's the weather like in Lisbon tomorrow?","language":"English"},"transcript_ms":311.4}
```

The turn's fields are `speaking`, `said` (the agent's reply so far), `unfinished`,
`speaker` (`caller` or `another_voice`; a participant's name never leaves the router) and
`conversation`. The turn goes in the body because the longest one a prompt shows, 800
characters of reply and six lines of 40 words, escapes to 10,285 bytes in Hindi and 12,035
in Russian as JSON in a header, past the 8 KiB a proxy allows one header line.

## How a ruling maps onto the flow controller

The server says what, and the router says when. `ruling` is what to do once the turn has
ended; with the floor free the router has it ended when `p_eot` reaches the bar for this
check, and otherwise rules wait and puts the same words again after `interval_ms`. The
first check of a pause is the first time its words are put (`FlowTurn.Waited` false), the
turn's first pause is one before any wait, and the pause runs from when the words last
changed.

| Model | `Decided.Disposition` | `Decided.Floor` | Also |
| --- | --- | --- | --- |
| floor free, `p_eot` under the bar and the pause under the cap | `wait` | `continue` | `Again`: `interval_ms` |
| `respond`, `clarify`, `ignore` | as said | `continue` | |
| `cannot` | `respond` | `continue` | `Cannot` |
| a tool | `respond` | `continue` | `Tool` |
| over the agent: `stop`, `shorten`, `continue` | `wait` while unfinished, else `respond` | as said | |
| over the agent: `ignore` | `ignore` | `continue` | |

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
  and a debug log line carries `p_end`, the bar, the pause, first check and first pause.
- **Transcripts.** A server that is down when the session starts, or that fails a
  request, is a fatal error, which hands the call to the next transcriber the target names.
  The words since the last ones written are lost. A target that names this transcriber
  alone has no next one.
- **A restart.** A 404 has the router register again and read `/v1/info` again, once.

## Measured

On an Apple M4 Max CPU, one lane, weights in f16, each including its request:

| | Median | Runs |
| --- | --- | --- |
| Decision, 2.8 s of audio | 111 ms | 5 |
| Decision, a full 16 s window | 639 ms | 5 |
| Decision and transcript, 2.8 s | 419 ms | 3 |
| Decision and transcript, 16 s | 1.47 s | 3 |

The transcriber, on a recorded "Hi, what's the weather like in Lisbon tomorrow?", wrote
"Hi." at the pause after it (880 ms of audio, in 239 ms), the whole question at its end
(2,680 ms, in 595 ms), and ended the utterance 2.5 s later.

A decision sends 512,000 bytes of PCM for a full window, plus the turn and the multipart
framing: 413 bytes with no conversation, and at most 64 KiB. Ogg Opus at 32 kb/s would be about 78,500
bytes, but the router's pure-Go encoder took 56 to 160 ms per 16 s window, more than a
GPU's whole decision, so the router sends PCM.

## Limits

- The pause is timed from the transcript's last change, which lags the caller's last sound
  by the transcriber's own delay. The policy was fitted to pauses timed from the voice.
- The transcriber's voice gate, 0.004 of full scale, was not measured on calls. It writes
  words only at pauses, so words said over the agent reach the flow controller at the
  caller's next pause. It rewrites the whole utterance at each pause, ends one at 60 s if
  the caller never pauses, and drops a ninth request waiting on a slow server.
- With both modes on, a pause costs two requests: the transcript asked for the default
  agent, and the decision asked for the agent's own.
- With unified replies the reply rules on its own words and no flow controller is asked,
  so acting has nothing to stand in for; shadow compares the model with the reply's ruling.
- A CPU decision over a full window, 639 ms here, does not fit the 250 ms an overlap
  gives it, so words over the agent fall back to the flow controller on such a server.

## Test it

```bash
ROUTER_AUDIO_FLOOR_URL=http://127.0.0.1:8080 AUDIO_FLOOR_WAV=caller.wav \
  go test -tags integration -run TestARealServer -v ./internal/audiofloor/
```
