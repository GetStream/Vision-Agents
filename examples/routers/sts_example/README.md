# STS example

Holding a conversation through a config this example writes itself, rather than one
somebody else wrote. `routers/frontdesk/router.yaml` asks for one native audio model, quick
to answer, that writes down what it hears and says, and naming the config stores it. The
clip next to this file is the caller; the model's reply is written next to it, because an
example has no speaker.

Nothing here names a model — the router picks — and nothing here transcribes, detects a
turn or speaks. The model on the other end of the socket does all three.

## The config

```yaml
sts:
  target: sts-fast
  input_transcript: true
  output_transcript: true
  data_policy:
    allow_training: false
```

`sts-fast` is a capability, not a model: the one quickest to first audio, pinned to Gemini
3.1 Flash Live while it is up and falling back to the rest of the tier when it is not. One
block where the cascade needs three, because there is no transcriber to keep in step with a
voice.

The two transcripts are terms rather than switches. One vendor cannot turn its transcript
off and two give only finals, so asking for them narrows the candidates to the models that
can, and a model that cannot is never asked. `data_policy` rules out the one vendor that has
published nothing about training: a vendor who has said nothing is not a vendor who has
said no.

What is missing is the interesting half. There is no `voice`, because a voice name belongs
to one vendor and the config does not name one; whichever answers uses its own. There is no
`turn_detection`, because only one of the four reads the words rather than waiting out a
pause, and asking for `semantic` would leave one candidate.

## The audio

`saturday_seven_thirty.wav`, eight seconds of somebody booking a table, streamed a chunk at
a time the way a call arrives and then two seconds of silence. The silence is not padding:
the model decides the caller has finished by hearing them stop, and a clip that ends the
instant the speech does never gives it that.

The model's voice comes back in binary frames, each opening with a header that says which
reply the chunk belongs to. That is how `stream.STS` drops the tail of a reply the caller
talked over: the model learns of a barge-in a round trip after the caller does, and the
chunks in that gap would otherwise play on.

## Run

A running acceleration router first: see [acceleration/README.md](../../../acceleration/README.md).
It needs `ROUTER_POSTGRES_DSN`, since a stored config is a row, and a key for whichever
model answers: `GOOGLE_API_KEY`, `OPENAI_API_KEY` or `XAI_API_KEY`. Provider credentials
live with the router, not with this example.

```bash
cd examples/routers/sts_example
uv sync
uv run sts_example.py
```

It prints what the model heard, what it said back, and how long the caller waited to hear
the reply begin, then writes `spoken.wav` with the reply in it:

```
heard: Hi, I'd like to book a table for four this Saturday at 7:30 patio if you have it, a high chair, and one of us has a peanut allergy.
said:  Got it—a table for four this Saturday at 7:30 on the patio, with a high chair and noting the peanut allergy.
  first audio 102ms after the caller stopped

spoken.wav: 7.0s at 24000Hz
```

That run is the routing doing its job. `sts-fast` prefers Gemini, whose project was over
its spending cap at the time, so the router recorded the refusal and started the next
candidate in the tier, Grok Voice, before the socket reported ready. The caller heard one
model answer and nothing of the one that did not. The hundred milliseconds is measured from
the last audio the example sent, silence included, to the first audio back; the model
decided the turn was over somewhere inside that silence, so the wait it felt was longer.

Nothing here calls `sync_routers`. `Router("frontdesk")` finds
`routers/frontdesk/router.yaml` and stores it on the first session; `.router_sync` records
the md5, so a run that edits nothing sends nothing.

The same session is what an `Agent` takes as its `llm`. Hand it
`router.sts.realtime()` and the agent runs no transcriber, turn detector or voice of its
own; the edge carries the call and the model holds the conversation.

Needs a `.env` with:

```
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```
