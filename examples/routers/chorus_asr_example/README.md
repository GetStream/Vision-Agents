# chorus-asr examples

chorus-asr is one speech model that hears the caller's audio: it can write what they said,
decide when they have finished, or both. Three short Go programs, one for each way of using
it. What each is and what the router needs is in
[acceleration/deploy/chorus-asr](../../../acceleration/deploy/chorus-asr/README.md).

| Program | Pick it when | Router | What it names |
| --- | --- | --- | --- |
| [`decide_and_transcribe`](decide_and_transcribe/main.go) | One model should hear the caller for both the words and the turn | `ROUTER_CHORUS_ASR_URL`, `ROUTER_CHORUS_ASR_MODE=act` | [`agents/ferry-concierge`](agents/ferry-concierge/agent.yaml): `stt: stream/chorus-asr`, its own `llm` and `tts` |
| [`decide_only`](decide_only/main.go) | You keep your transcriber, and take the turn-taking from the audio | `ROUTER_CHORUS_ASR_URL`, `ROUTER_CHORUS_ASR_MODE=act` | [`agents/clinic-front-desk`](agents/clinic-front-desk/agent.yaml): `stt: deepgram/flux-general-en`, the server's default `end_of_turn` |
| [`transcribe_only`](transcribe_only/main.go) | You want the words alone, and your own pipeline decides when to answer | `ROUTER_CHORUS_ASR_URL` (the mode stays `off`) | [`routers/chorus-asr.yaml`](routers/chorus-asr.yaml): `stream/chorus-asr` as the router config's one STT provider |

## Run

A router with a chorus-asr server (see [acceleration/README.md](../../../acceleration/README.md)),
started with the settings the program needs:

```bash
ROUTER_CHORUS_ASR_URL=http://floor:8080 ROUTER_CHORUS_ASR_MODE=act   # the first two
ROUTER_CHORUS_ASR_URL=http://floor:8080                              # transcribe_only
```

`ROUTER_CHORUS_ASR_MODE=shadow` runs the first two with chorus-asr asked beside the flow
controller and acted on for nothing, to compare the two on real calls first.

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

## Choosing how eager turn-taking is

The two agents end their turns by the server's default. An agent chooses otherwise with
`end_of_turn` on its config (`POST /v1/agents/sync`, `PATCH /v1/agents/configs/{id}`), or a
server-side `createSession` for one session: one of `max_wait_ms` (the measured point with the
fewest cut-ins that answers within it at p90), `preset` (one of the server's, such as `safe`
or `fast`) or `bars` of your own. A wait under the quickest point's p90 is refused, and one
whose point fails the dictation guard is kept with a warning saying what it does to callers
reading out numbers. The trade-off each point makes is in
[acceleration/deploy/chorus-asr](../../../acceleration/deploy/chorus-asr/README.md#how-eager-turn-taking-is-end_of_turn).
