# Configuration and lifecycle

Add the installed plugin to Mesh configuration, followed by model entries from the individual guides:

```toml
[[plugin]]
name = "onnx"
command = "/absolute/path/to/mesh-onnx"
```

Then start `mesh-llm serve`. `command` can be omitted when Mesh resolves the installed plugin package. Model-selection arguments and a standalone downloader are unnecessary. Inspect the node's model discovery/status to find the configured model and its state.

Each model has `model` (the exact repository and revision), `provider = "onnx"`, optional `variant = "fp32" | "int8"`, `[models.options]`, `[models.request_defaults]`, `[models.lifecycle]` and `[models.source]`. Indra supports its published FP32 bundle; the other three support FP32 and pre-exported INT8 variants. Unknown options and incompatible revisions fail before acquisition. One serving ID may only be configured once per process; simultaneous variants require separate serving identities in a future extension.

Model options include `threads` (1–256, default 4). Indra additionally accepts `math`. Request defaults use each adapter's schema; a multipart request may override a default within hard bounds. PDF passwords and CLIP labels are request-only inputs and cannot be persisted as defaults. Incompatible controls, including OCR `dpi` on a transcription request, are rejected.

```toml
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

`load = "startup"` loads the engine during startup preparation; `on_demand` creates sessions on first use. `prefetch = true` acquires files at startup even for on-demand models, without creating sessions. With `false`, acquisition waits for use. Idle eviction drops sessions after the configured seconds without admitted or active work, retaining the disk cache. `idle_timeout_secs = 0` disables eviction. Defaults are startup loading, prefetch enabled, idle eviction disabled.

Concurrent requests share a loaded instance and one load attempt. Per-model admission is bounded at 32 requests; ONNX sessions execute serially per configured model. Idle eviction cannot drop an engine held by an active or queued request. OCR requests carry a per-request acquisition/loading/inference deadline: `deadline_secs` (30–900, multipart field), defaulting to `min(60 + 2 * selected_pages, 600)`. A cold request can therefore time out while the host's acquisition job continues; a later attempt can reuse the cache. The timeout is a `408`-style error naming the budget and suggesting `pages` for chunking. Transcription, similarity and depth keep a flat 120-second deadline. Whisper cancellation is checked during token generation. Native ORT calls themselves cannot be interrupted mid-call.

Cold configured models remain listed as `loadable`; `ready` means resident and `busy` means the model mutex is held. Plugin process health is separate from residency. Actual RSS may remain above the startup baseline because the runtime or allocator caches memory.

## Existing local assets

To avoid all network acquisition, add a complete model bundle directory:

```toml
[models.source]
local_dir = "/absolute/path/to/model-bundle"
```

Keep the upstream relative structure: non-Indra graphs live under `onnx/`, while tokenizer and processor metadata live at the bundle root. Depth requires the selected graph's sibling `.onnx_data` file. Model recipes request exact required files, not all repository variants. Source repository/revision overrides are also supported for a compatible export; changing a source does not change the adapter's tensor contract.

## Upload cap

Uploads are capped at 100 MiB by default. Set `ONNX_MAX_BYTES` (bytes) in the plugin process environment or pass the `--max-bytes` plugin flag to change it; a body over the cap fails with `413` and a message naming the limit. Uploads are spooled to a temp file under the OS temp directory (prefix `mesh-onnx-`) rather than buffered in memory, and removed when the request completes. The effective cap is reported per model as `max_upload_bytes` in `GET /v1/models`.

## HTTP representations

Use `Accept: application/json`. OCR also supports `text/plain`, `text/markdown` and `application/x-ndjson` (canonical route only, streamed); transcription also supports `text/plain`; depth also supports `image/png`. JSON is the default, and an absent or empty `Accept` means no preference rather than an error. Unsupported representations return 406. Responses include `Content-Type` and `Vary: Accept`. `response_format` is not an accepted form field on these native task surfaces. Any separate OpenAI compatibility projection must translate its protocol controls explicitly.

Requests are multipart with required `model` and `file`. Let curl set `Content-Type` and the multipart boundary. Uploads are capped as described above; control fields are bounded. Task defaults and input semantics are documented in each model guide. Errors are JSON objects with `error.message` that name what was asked, the limit, and the remedy.
