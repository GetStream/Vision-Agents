# chorus-asr examples

chorus-asr is one speech model that hears the caller's audio: it can write what they said,
decide when they have finished, or both. Three short Go programs, one for each way of using
it. What each is and what the router needs is in
[acceleration/deploy/chorus-asr](../../../acceleration/deploy/chorus-asr/README.md).

| Program | Pick it when | Router | What it names |
| --- | --- | --- | --- |
| [`decide_and_transcribe`](decide_and_transcribe/main.go) | One model should hear the caller for both the words and the turn | `ROUTER_CHORUS_ASR_URL` | [`agents/ferry-concierge`](agents/ferry-concierge/agent.yaml): `floor_control: {model: stream/chorus-asr, max_wait_ms: 1100}`, `stt: stream/chorus-asr`, its own `llm` and `tts` |
| [`decide_only`](decide_only/main.go) | You keep your transcriber, and take the floor control from the audio | `ROUTER_CHORUS_ASR_URL` | [`agents/clinic-front-desk`](agents/clinic-front-desk/agent.yaml): `floor_control: {model: stream/chorus-asr}`, `stt: deepgram/flux-general-en`, the server's default point |
| [`transcribe_only`](transcribe_only/main.go) | You want the words alone, and your own pipeline decides when to answer | `ROUTER_CHORUS_ASR_URL` | [`routers/chorus-asr.yaml`](routers/chorus-asr.yaml): `stream/chorus-asr` as the router config's one STT provider |

## Run

A router with a chorus-asr server (see [acceleration/README.md](../../../acceleration/README.md)),
which makes chorus-asr available to every agent on it and rules none of their floors itself:

```bash
ROUTER_CHORUS_ASR_URL=http://floor:8080
```

Each agent opts in to chorus-asr's floor control in its own `agent.yaml`, so one router serves
agents with it and without it at once:

```yaml
floor_control:            # leave it out for the router's default; {model: flow-controller} keeps the flow controller
  model: stream/chorus-asr
  max_wait_ms: 1100       # optional: how long a caller may wait once they stop
```

A router without `ROUTER_CHORUS_ASR_URL` refuses the sync of an agent that names it, saying so.

Then point the SDK at the router and run one:

```bash
export STREAM_ACCELERATION_URL=http://localhost:<port> STREAM_ACCELERATION_CUSTOMER_ID=acme
cd decide_and_transcribe && go run .
```

The two agent programs sync their folder under `agents/`, join a Stream call and print a link
to talk on it, then the conversation. `transcribe_only` syncs `routers/` and streams the
realtime STT example's clip through it.

Transcribing alone, the router does not know what your pipeline says, so anything of your
own voice that comes back down the caller's line is written as words, as with any STT: a
final of its own after the caller's. Cancel the echo at the edge, on the device or the media
server, as a loudspeaker needs for every transcriber. The two agent programs need no such
care from you for the router's own agent: it drops its own voice before the conversation
hears it.

## Choosing how long a caller waits

`floor_control.max_wait_ms` is the one knob: the most a caller who has finished may wait for
the answer. It chooses the measured point with the fewest cut-ins that answers within it at
p90; the ferry agent waits at most 1.1 s, and the clinic agent, naming none, takes the
server's default point. A wait under the quickest point's p90 is refused, and one whose point
fails the dictation guard is kept with a warning saying what it does to callers reading out
numbers. The trade-off each point makes is in
[acceleration/deploy/chorus-asr](../../../acceleration/deploy/chorus-asr/README.md#how-long-a-caller-waits-max_wait_ms).
