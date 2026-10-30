# Validation

## Offline checks

Dependencies, including the pinned Mesh SDK and ONNX Runtime binaries, must
be fetched before running offline. The ignored tests below are separate
real-model checks and require the documented local bundles.

```sh
cargo fmt --check
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
```

These cover configuration/task isolation, bounded request overrides, `Accept`
negotiation, unloaded-model HTTP discovery and health, multipart rejection,
residency leases/eviction/cancellation, preprocessing, token handling, decoder
control, relative-depth preview and PDF text geometry. They download nothing.

Transcription additionally covers OpenAI's `response_format` field: `json` and
`text` are accepted as the shaping mechanism an OpenAI SDK uses, while
`verbose_json`, `srt` and `vtt` are refused because this decoder produces no
timestamps. A refusal there happens before weights are touched.

The real Pdfium text-layer test needs no weights: set `PDFIUM_LIBRARY` to a local
platform library and run the ignored test.

## Real-model checks

Every weight-dependent test is `#[ignore]`d and reads a local bundle from an
environment variable. Acquire the pinned bundles through Mesh's shared asset
service, or provide complete local directories via `[models.source]`. Never
download a whole repository snapshot — the recipes name the exact files.

| Variable | Task | Variant |
|---|---|---|
| `ONNX_OCR_MODEL_DIR` | Indra OCR | (no quantized variant) |
| `ONNX_CLIP_MODEL_DIR` / `ONNX_CLIP_INT8_MODEL_DIR` | Image/text similarity | FP32 / INT8 |
| `ONNX_WHISPER_MODEL_DIR` / `ONNX_WHISPER_INT8_MODEL_DIR` | Transcription | FP32 / INT8 |
| `ONNX_DEPTH_MODEL_DIR` / `ONNX_DEPTH_INT8_MODEL_DIR` | Relative depth | FP32 / INT8 |
| `ONNX_WHISPER_WAV` | Transcription input | 16 kHz mono, ≤30 s |
| `PDFIUM_LIBRARY` | PDF input | native library |

A test skips a variant whose variable is unset and fails if none is set, so a
typo in a variable name cannot turn into a silent pass.

```sh
cargo test --offline --lib -- --include-ignored
```

Because `--include-ignored` runs the whole suite, run it with all variables
exported; otherwise the weight-dependent tests fail on the missing variable
rather than skipping.

## What the weight tests assert

Passing a session-construction check is not an inference pass. Each test pins a
result, not just that the graph loaded:

- **CLIP** — both encoders produce 512-dimension vectors, and ranking the
  scanned fixture against three labels puts the text label first at a score
  above 0.2. With both variants present, the INT8 ranking must agree label for
  label within 0.1 cosine.
- **Whisper** — a spoken fixture transcribes to the expected phrases with
  `language = "en"` from auto-detection, and a second pass over the same audio
  reproduces the transcript exactly. That second pass is the request-local
  decoder-state check: a leaked KV cache drifts.
- **Depth** — the graph runs from its external-data file, produces finite
  relative values, and renders a preview at the map's own dimensions. With both
  variants present, FP32 and INT8 maps must correlate above 0.9 — the INT8
  export uses 4-bit `MatMulNBits`, so this is also the contrib-ops check.
- **Indra** — the scanned fixture yields its expected 35 lines, 8 layout blocks
  and the exact Korean/English text, through native ORT sessions.
- **PDF** — `mode = "text"` reads the text layer without a model session and
  scales coordinates with DPI; `mode = "auto"` on a scan with no text layer
  renders the page and reports `source = "ocr"`.

## Reported status

All of the above pass on macOS arm64 with the pinned revisions, through `ort`
2.0.0-rc.12, in both FP32 and INT8 (INT8 for CLIP, Whisper and Depth):

```text
cargo test --offline --lib -- --include-ignored
test result: ok. 56 passed; 0 failed; 0 ignored
```

Not covered by an automated test yet: Indra math (`options.math = true`), which
adds five more graphs and a LaTeX decoder; encrypted PDFs; the hybrid page path
(text layer plus image regions, `keep_furniture`); and any multi-node routing.
Those remain manual checks.

## Mesh-level checks

Run the README HTTP workflows against Mesh, including a second node routing to
each cold provider. Verify independent loading, concurrent request isolation,
deadline/cancellation behavior, idle eviction and offline cached reload. Keep
fixture artifacts cached between runs. Record exact model revision, variant,
runtime and result tolerances with the validation report.

### Single-node pass, recorded

All four models were driven through Mesh's inference listener (`:9337`), not the
plugin's own port, with the plugin declared by `command` and every model set to
`load = "on_demand"`, `idle_timeout_secs = 300`, `prefetch = false`, serving
from local `[models.source]` directories:

| Request | Result |
|---|---|
| `POST /mesh/v1/documents/ocr` — PNG | 200 · 2.93 s · 35 lines · `source: ocr` |
| `POST /mesh/v1/documents/ocr` — PDF | 200 · 6.67 s · 35 lines · `source: ocr` |
| `POST /v1/audio/transcriptions` — 16 kHz WAV | 200 · 1.85 s · 22 tokens · `language: en` |
| `POST /v1/audio/transcriptions` — `response_format=text` | 200 · `text/plain` transcript, as an OpenAI client asks for it |
| `POST /v1/audio/transcriptions` — `response_format=srt` | 400 · refused with the reason, not answered without timings |
| `POST /mesh/v1/images/similarity` — PNG + 3 labels | 200 · 1.28 s · text label first, 0.2817 |
| `POST /mesh/v1/images/depth`, `Accept: application/json` | 200 · 2.97 s · 1653×2337 · `relative` |
| `POST /mesh/v1/images/depth`, `Accept: image/png` | 200 · 2.05 s · 1653×2337 8-bit grayscale |
| OCR with `Accept: image/png` | 406 |
| OCR with no `model` field | 400 |

Idle eviction was checked with `idle_timeout_secs = 15`: after one
transcription and a 25 s wait, all four models reported `loadable` on the
plugin's `/v1/models` (sessions dropped, weights kept); the next request
reloaded in 1.86 s and reproduced the same transcript exactly.

Two behaviours that look like faults and are not:

- Mesh's `/v1/models` returns an empty list on a node whose models are all task
  endpoints. Mesh lists only chat-protocol models there, and deliberately does
  not advertise a document-OCR or depth model as chat. The plugin's own
  `/v1/models` lists all four with `availability` and `workload_class`.
- A `406` is an ordinary outcome of `Accept` negotiation, not a provider
  failure. Mesh relays it with the original status, so a client can correct the
  request instead of retrying.

Not yet exercised this way: a second node routing to each cold provider, and
concurrent request isolation across two callers. Those remain manual checks.
