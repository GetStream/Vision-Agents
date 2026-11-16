---
name: router-detection
description: What the detection router can be asked for, what each vendor calls it, and what it refuses to fake. Read before adding a detection option or a detection provider (Moondream, Roboflow, Ultralytics YOLO).
---

# Routing detection

The per-modality half of [router-interface](../router-interface/SKILL.md). The contract is
[`internal/detection`](../../../acceleration/internal/detection/detection.go), the router is
[`internal/detectionrouter`](../../../acceleration/internal/detectionrouter), the providers sit
under `internal/detection/<vendor>`, and they are declared in the `detection:` section of
[`router.yaml`](../../../acceleration/internal/routing/router.yaml).

Detection is one image in and boxes out, so like search and image generation it is a plain
`POST /v1/detection` with no socket. The router never sees the call's video: a worker (the
Python `stream.DetectionProcessor`) samples frames and sends them one at a time. One request is
one stat row with `images: 1`, priced `per_image`.

Every box comes back normalized to 0..1 as `x_min, y_min, x_max, y_max`, whatever the vendor
sent, so a caller never needs the image size to draw it.

## The vendors, and what each calls the same thing

| Option | Moondream | Roboflow model | Roboflow Workflow | Ultralytics YOLO |
| --- | --- | --- | --- | --- |
| endpoint | `POST api.moondream.ai/v1/detect` | `POST serverless.roboflow.com/{project}/{version}` | `POST serverless.roboflow.com/workflows/run` or `/infer/workflows/{ws}/{id}` | `POST platform.ultralytics.com/api/models/{owner}/{project}/{model}/predict` |
| auth | `X-Moondream-Auth` | `api_key` query | `api_key` body | `Authorization: Bearer`, optional for public models |
| image | `image_url` data URI | base64 body, form encoded | `inputs.image {type: base64}` | multipart `file` |
| `classes` | `object`, free text, one call per class | filter after the fact | filter after the fact | filter after the fact |
| `confidence` | none, no score returned | `confidence` | inside the Workflow | `conf` |
| box | normalized `x_min..y_max` | centre `x, y, width, height` in pixels | same as a model | `x1..y2`, normalized with `normalize=true` |
| embeddings | none | none | what the Workflow outputs as `embeddings` | none |

What the table is saying:

- **Only Moondream is open vocabulary.** It finds whatever noun it is asked for, so "package" or
  "a red mug" works. A Roboflow or YOLO model has the classes it was trained on (COCO's 80 for
  `coco/3` and `yolo26n`), and asking one for a class it does not have finds nothing.
- **Moondream returns no confidence.** Its boxes carry `confidence: 1`, and a request asking
  for a confidence floor is not routed to it.
- **Embeddings come from a Workflow.** `roboflow/faces` is a built-in Workflow (face detection,
  dynamic crop, CLIP) whose boxes each carry a 512-dimension vector. CLIP describes how a crop
  looks, not who it is: it re-identifies the same person in the same light and clothes, and is
  not face identification.

## What the router refuses to fake

The terms a detection model declares in `supports:`:

| Term | Meaning | Declared by |
| --- | --- | --- |
| `open_vocabulary` | `classes` may be any words, not only trained labels | Moondream |
| `confidence` | `min_confidence` is honoured | Roboflow, Ultralytics |
| `embeddings` | each box carries a vector | `roboflow/faces`, `roboflow/workflow` |

A request that asks for one is only routed to a model that declared it, and a term nothing can
serve is a 400 naming it. A fixed-class model asked for "package" would return an empty list
that looks like "there is no package", which is the exact failure terms exist to prevent.

## Overwrites

`options.overwrites` carries what the vocabulary has no word for, keyed by provider:

```json
{"overwrites": {"roboflow": {"workspace": "acme", "workflow_id": "doorbell"}}}
```

- `roboflow/workflow` needs `workspace` and `workflow_id` (a saved Workflow), or `specification`
  (an inline one). It must output `predictions`, and may output `embeddings`, one per box.
- There is no overwrite for an address. Ultralytics' shared API is limited to 20 requests a
  minute, and a deployment that needs more names a dedicated endpoint in `ULTRALYTICS_ENDPOINT`.
  That is the operator's to set, because whatever address it names is sent the key and the
  frame.

## Adding a provider

1. A package under `internal/detection/<vendor>` implementing `detection.Provider`, reading its
   key from the environment, normalizing boxes, and failing with `detection.Fail`.
2. Register it in `detectionrouter.DefaultRegistry`.
3. Declare its models in `router.yaml` with `per_image` and the terms it really sends.
4. A unit test against an `httptest` server for the request it sends and the boxes it reads, and
   an integration test against the real API.
