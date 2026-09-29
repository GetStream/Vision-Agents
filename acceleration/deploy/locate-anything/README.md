# LocateAnything-3B on Baseten

NVIDIA's [LocateAnything](https://research.nvidia.com/labs/lpr/locate-anything/), an
open-vocabulary grounding model, behind an OpenAI-compatible `/v1/chat/completions` so the
LLM router's `locateanything` provider can reach it. Shown an image and a description, it
answers with boxes:

```
<ref>car</ref><box><12><140><58><190></box><box><64><141><104><188></box>
```

Coordinates are on a 0 to 1000 grid over the image, whatever its size.

Live on Baseten as `locate-anything-3b` (`q40g82yw`), one H100, scaling to zero after 15
idle minutes:

```bash
LOCATE_ANYTHING_BASE_URL=https://model-q40g82yw.api.baseten.co/environments/production/sync/v1
```

## Licence

The weights are under the **NVIDIA License for non-commercial use**: academic and
non-profit research only. This deployment is for demos and evaluation, and nothing that
bills a customer should route to it.

## Why a model.py

The checkpoint ships its own modeling code (`trust_remote_code`) and decodes boxes in
parallel blocks, which neither vLLM nor TensorRT-LLM run. So this is plain Transformers in
`model/model.py`, exposed through Truss's `chat_completions` hook.

The model decodes an answer whole, so a streamed response is one content chunk followed by
a usage chunk. Only the last user message is read: its first `image_url` part (a data URI
or an http URL) and its text.

## Deploy

```bash
truss push --promote
```

Then point `LOCATE_ANYTHING_BASE_URL` at the OpenAI-compatible root:

```bash
LOCATE_ANYTHING_BASE_URL=https://model-$MODEL_ID.api.baseten.co/environments/production/sync/v1
```

`BASETEN_API_KEY` is the bearer token. If Hugging Face asks for the licence to be accepted
before download, accept it on the model page and set the `hf_access_token` secret in Baseten.

## Prompts

| Task | Text |
| --- | --- |
| Detection | `Locate all the instances that matches the following description: car</c>truck.` |
| One instance | `Locate a single instance that matches the following description: the red car.` |
| Pointing | `Point to: the exit.` |

`GENERATION_MODE` picks `fast`, `slow` or `hybrid` (the default); a request may override it
with a `generation_mode` field.

Ask for one category per request when the scene is dense. Asked for `parked car</c>empty
parking space` together on a full lot, the model found the cars and then repeated one
empty space until it ran out of tokens; asked separately, each came back in two to five
seconds. Hybrid mode at the default temperature of 0.7 found 41 or 42 cars on the same
photo run after run. Greedy decoding (`temperature: 0`) is not steadier: hybrid mode
occasionally ends an answer early either way. Slow mode is steady but takes about twice as
long and finds fewer cars.

## Test it

```bash
curl -s $LOCATE_ANYTHING_BASE_URL/chat/completions \
  -H "Authorization: Api-Key $BASETEN_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"LocateAnything-3B","messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/d/df/Aerial_view_of_an_Abuja_parking_lot.jpg"}},
        {"type":"text","text":"Locate all the instances that matches the following description: car."}]}]}'
```

Or through the router, with the parking lot demo in `sdks/go/examples/parking`.
