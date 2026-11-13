# Using each model through Mesh over HTTP

Mesh serves these on its inference port (default `9337`). The plugin's own
listener is an ephemeral loopback port; requests go to Mesh, which admits the
request by workload and forwards it to a node serving the model.

Every request is `multipart/form-data` and needs **two** fields on every task:

- `model` — the exact string configured in `[[models]]`, including the
  `@revision`. It is required; a name that is not configured is a `400`, and
  selecting a model that does not serve the endpoint's task is a `400` too.
- `file` — the input. Uploads are capped at 100 MiB by default; set
  `ONNX_MAX_BYTES` or the plugin's `--max-bytes` flag to change it, and read
  the effective cap per model as `max_upload_bytes` from `GET /v1/models`.

Anything else is task-specific and may be omitted when
`[models.request_defaults]` supplies it. Let curl set the multipart
`Content-Type` and boundary.

| Task | Endpoint | Protocol | Accept |
|---|---|---|---|
| Document OCR | `POST /mesh/v1/documents/ocr` | `document_ocr_v1` | `application/json`, `text/plain`, `text/markdown`, `application/x-ndjson` |
| Transcription | `POST /v1/audio/transcriptions` | `transcription_v1` | `application/json`, `text/plain` |
| Image/text similarity | `POST /mesh/v1/images/similarity` | `image_text_similarity_v1` | `application/json` |
| Relative depth | `POST /mesh/v1/images/depth` | `depth_estimation_v1` | `application/json`, `image/png` |

JSON is the default. A type the endpoint cannot produce is `406`, and every
response carries `Vary: Accept`. OCR's `application/x-ndjson` is canonical-route
only and streams events as pages complete (see
[Streaming OCR results](#streaming-ocr-results-ndjson)); the legacy
`/v1/documents/ocr` alias keeps the v0.1 representations and never streams, and
the v0.2 fields (`pages`, `pages_only`, `stream`, `deadline_secs`) are rejected
there. `response_format` is not accepted on the
`/mesh/v1/` task routes, whose options are body fields of their own. The
OpenAI-shaped transcription route is the exception: it accepts
`response_format=json|text`, because that is how an OpenAI client names the
body there, and refuses `verbose_json`, `srt` and `vtt` with a reason rather
than answering without the timings they promise.

## Two path namespaces

`/v1/` is reserved for OpenAI-shaped APIs, so an OpenAI client or SDK pointed at
Mesh finds what it expects and Mesh never squats a path OpenAI may take later.
Mesh's own task contracts live under `/mesh/v1/` with the same versioning
convention:

| Namespace | What belongs there |
|---|---|
| `/v1/…` | OpenAI-shaped: `chat/completions`, `completions`, `responses`, `embeddings`, `rerank` (Cohere/Jina/vLLM use the same path, so it is a de-facto external standard), `audio/transcriptions`, `audio/speech`, `realtime`, `decisions`, `models` |
| `/mesh/v1/…` | Mesh tasks with no external equivalent: `documents/ocr`, `images/similarity`, `images/depth` |

The rule matters operationally: Mesh admits a request by matching its path to a
workload class **and** forwards that same path to the plugin, so the string is
written down in both repositories. Renaming it in one place and not the other
fails at runtime with a `404`, never at compile time. Treat these paths as a
two-sided contract.

## Configuration

All four models, one plugin process:

```toml
[[plugin]]
name = "onnx"
command = "/absolute/path/to/mesh-onnx"

[[models]]
model = "datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b"
provider = "onnx"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true

[[models]]
model = "onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7"
provider = "onnx"
variant = "int8"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true

[[models]]
model = "Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523"
provider = "onnx"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true

[[models]]
model = "onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e"
provider = "onnx"
variant = "int8"

[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

The `model` value above is also the `model` field value in every request — there
is no separate serving name. A configured model that is not resident still
appears as `loadable`; the first request loads it, and the OCR deadline
(`deadline_secs`, default `min(60 + 2 * selected_pages, 600)`) covers
acquisition, loading and inference, so a cold start on a slow link can
time out while the host's download continues. A later request reuses the cache.

To stay off the network entirely, add `[models.source] local_dir = "..."` with
the bundle laid out as the upstream repository is (`onnx/` inside, metadata at
the root).

## Document OCR

```sh
curl --fail-with-body \
  -H 'Accept: application/json' \
  -F 'model=datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b' \
  -F 'file=@scan.pdf' \
  http://localhost:9337/mesh/v1/documents/ocr
```

| Field | Values | Default |
|---|---|---|
| `dpi` | 36–600 | 200 |
| `mode` | `auto`, `ocr`, `text` | `auto` |
| `image_ocr` | `true`/`false` — OCR images embedded in a text-layer PDF | `true` |
| `keep_furniture` | `true`/`false` — keep headers and footers in Markdown | `false` |
| `max_pages` | 1–1000 — cap the pages processed | 1000 |
| `pages` | `3`, `1-10`, `1-` — select a 1-based inclusive range; mutually exclusive with `max_pages`; out-of-range is a `400` naming the real page count | — |
| `pages_only` | `true`/`false` — metadata only: page count and rendered `page_sizes`, no model session, works on a cold model | `false` |
| `stream` | `true`/`false` — stream NDJSON page events (canonical route only; `Accept: application/x-ndjson` also selects it) | `false` |
| `deadline_secs` | 30–900 — per-request acquisition + inference budget | `min(60 + 2 * selected_pages, 600)` |
| `password` | opens an encrypted PDF; request-only, never a default | — |

Booleans and numbers are sent as JSON literals (`-F 'image_ocr=false'`,
`-F 'dpi=300'`); `mode`, `pages` and `password` are strings. PDF input needs
Pdfium on the plugin host; images do not.

```json
{"model":"datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b","page_from":1,"page_to":1,"pages":[{"page_index":0,"text":"Invoice 4471","markdown":"Invoice 4471","source":"ocr","rotation":0,"vertical":false,"lines":[{"text":"Invoice 4471","bbox":[10,20,150,40],"confidence":0.95}],"blocks":[]}]}
```

`page_from`/`page_to` are 1-based inclusive document bounds, so a chunked
caller can place each response next to its neighbors. `text/plain` and
`text/markdown` return the concatenated page text, separated by a blank line.

## Streaming OCR results (NDJSON)

On the canonical route, `Accept: application/x-ndjson` (preferred) or the
multipart field `stream=true` selects a streaming response:
`Content-Type: application/x-ndjson`, one JSON object per line, emitted as
each page completes:

```sh
curl --fail-with-body -N -H 'Accept: application/x-ndjson' \
  -F 'model=datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b' \
  -F 'file=@paper.pdf' -F 'pages=1-10' \
  http://localhost:9337/mesh/v1/documents/ocr
```

```json
{"type":"meta","model":"datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b","pages":10,"page_from":1,"page_to":10}
{"type":"page","page":1,"text":"…","markdown":"…","lines":[…],"blocks":[…]}
{"type":"page","page":2,"text":"…","markdown":"…","lines":[…],"blocks":[…]}
{"type":"done","pages":10}
```

`meta` is always first, `page` events arrive in page order as pages finish
(`page` is the 1-based document page number), and `done` is last. A failure
after streaming started is delivered as
`{"type":"error","error":{"message":"…"}}` and ends the stream; failures
before the first page (negotiation, validation, model load) stay normal HTTP
error statuses with the JSON error body, not a `200` stream. `pages_only=true`
answers with plain metadata JSON instead of a stream. The legacy
`/v1/documents/ocr` alias never streams: NDJSON is a `406` there and
`stream=true` is rejected.

## Transcription

```sh
curl --fail-with-body \
  -H 'Accept: text/plain' \
  -F 'model=onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7' \
  -F 'file=@speech.wav' \
  http://localhost:9337/v1/audio/transcriptions
```

| Field | Values | Default |
|---|---|---|
| `language` | `auto` or a 2–3 letter lowercase code | `auto` |
| `max_output_tokens` | 1–444 | 128 |

WAV, mono or stereo, PCM16 or float32, up to 30 seconds; it is resampled to
16 kHz. Longer audio is rejected rather than truncated, because this provider
does one window and does not chunk. JSON returns `text`, `language` and
`generated_tokens`.

## Image/text similarity

```sh
curl --fail-with-body \
  -H 'Accept: application/json' \
  -F 'model=Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523' \
  -F 'file=@photo.jpg' \
  -F 'labels=["a photo of a dog","a photo of a cat","a diagram"]' \
  http://localhost:9337/mesh/v1/images/similarity
```

| Field | Values | Default |
|---|---|---|
| `labels` | JSON array, 1–100 non-empty texts; **required per request**, cannot be a default | — |
| `top_k` | 1–100 | 3 |
| `truncate_text` | `true`/`false` | `false` |

```json
{"model":"Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523","scores":[{"label":"a photo of a dog","score":0.91}],"score_type":"cosine_similarity"}
```

Scores are cosine similarities, ranked, cut to `top_k`. This endpoint ranks
against labels; it does **not** return a bare image embedding, so there is no
route for image vectors over Mesh today.

## Relative depth

```sh
curl --fail-with-body \
  -H 'Accept: application/json' \
  -F 'model=onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e' \
  -F 'file=@scene.jpg' \
  http://localhost:9337/mesh/v1/images/depth
```

| Field | Values | Default |
|---|---|---|
| `output_resolution` | `input` (upload's grid) or `native` (model grid) | `input` |
| `visualization` | `none` or `normalized` (adds a JSON `preview`) | `none` |

JSON returns `model`, `width`, `height`, `units: "relative"` and a row-major
`values` array — relative depth, not metres, and not comparable across images.

For a viewable image, ask for PNG instead, which is the preview and ignores
`visualization`:

```sh
curl --fail-with-body \
  -H 'Accept: image/png' \
  -F 'model=onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e' \
  -F 'file=@scene.jpg' \
  -o depth.png http://localhost:9337/mesh/v1/images/depth
```

## Discovery

Plugin-only tasks are deliberately kept out of the chat-compatible `/v1/models`
listing, so use the configured `model` string rather than expecting these models
to appear in an OpenAI-style model list. Mesh's plugin status view shows each
configured model, its variant and its residency (`loadable`, `ready`, `busy`).

## Common failures

| Status | Cause |
|---|---|
| `400` | `model` missing or not configured, a model asked to serve the wrong task, an unknown or out-of-range field, bad `pages` syntax, a `pages`/`max_pages` combination, or input the task rejects |
| `401` | An encrypted PDF opened without the `password` field (including `pages_only=true`) |
| `406` | `Accept` names no representation this endpoint produces |
| `413` | Upload is empty (400) or over the configured cap, default 100 MiB (`ONNX_MAX_BYTES`) |
| `503` | Model could not be loaded, plugin unreachable, or Pdfium missing for PDF input |
| `408` | The per-request deadline expired before the model answered; the message names the budget and suggests `pages` for chunking |

## Worked example: one claim submission, four models

A field technician finishes a job and submits three things from the phone app: a
photo of the damaged unit, a scanned service form, and a short voice note. The
backend runs them through Mesh in one pipeline.

```sh
BASE=http://localhost:9337
OCR_MODEL=datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b
CLIP_MODEL=Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523
DEPTH_MODEL=onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e
WHISPER_MODEL=onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7
```

**1. Read the form.** `mode=auto` uses the text layer when the PDF has one and
falls back to recognizing pixels. With the default `image_ocr=true`, embedded images can still be recognized; use `image_ocr=false` to skip those regions or `mode=text` to extract only text without a model session.
`source` in the response says which path was taken.

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F "model=$OCR_MODEL" -F 'mode=auto' -F 'max_pages=3' \
  -F 'file=@service-form.pdf' "$BASE/mesh/v1/documents/ocr"
```

A longer form can be chunked instead of capped: `pages=1-10`, then
`pages=11-20`, reassembling by the response's `page_from`/`page_to`, or read
straight off an NDJSON stream with `-H 'Accept: application/x-ndjson'`, which
delivers each page as it completes. `pages_only=true` on the first upload
reports the page count and rendered sizes before committing to a pass.

**2. Route the photo.** Classification is zero-shot against labels the backend
supplies, so the taxonomy lives in the application rather than in the model:

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F "model=$CLIP_MODEL" -F 'file=@damage.jpg' -F 'top_k=2' \
  -F 'labels=["hail damage","impact damage","water damage","normal wear"]' \
  "$BASE/mesh/v1/images/similarity"
```

**3. Judge whether the photo is usable.** Depth is not a measurement: it says
which parts of the frame are near and which are far, in relative units. That is
enough to flag a photo taken so close that the damage fills it, or one taken
across a room where the detail is too small to assess.

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F "model=$DEPTH_MODEL" -F 'file=@damage.jpg' \
  -F 'output_resolution=native' "$BASE/mesh/v1/images/depth"
```

For a human reviewer, store the PNG rather than the JSON:

```sh
curl --fail-with-body -H 'Accept: image/png' \
  -F "model=$DEPTH_MODEL" -F 'file=@damage.jpg' -o damage-depth.png \
  "$BASE/mesh/v1/images/depth"
```

**4. Transcribe the voice note.** The window is 30 seconds, so the app either
caps the recording or splits it and joins the text:

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F "model=$WHISPER_MODEL" -F 'file=@note.wav' -F 'language=auto' \
  "$BASE/v1/audio/transcriptions"
```

The claim record then holds the form text, the damage class, a documentation
quality flag and the technician's own words — four models behind one port. Uploads travel to the serving node in a multi-node mesh; keep the mesh on trusted nodes. Repeat submissions hit the cache and a resident model
instead of paying load time again.
