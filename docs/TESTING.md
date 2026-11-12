# Validation

## Local checks

Use Rust 1.98 or newer. Fetch dependencies, including the pinned Mesh SDK and
ONNX Runtime, before running offline:

```sh
cargo fmt --all --check
cargo test --locked --offline
cargo clippy --locked --offline --all-targets -- -D warnings
```

The default suite needs no model weights or Pdfium. Opt-in tests fail when their
required fixture variables are missing; they never silently certify absent models.

## Real model and PDF fixtures

Provide the exact recipe bundles, preserving upstream relative filenames.
Acquire named assets rather than whole repository snapshots.

| Variable | Bundle |
|---|---|
| `ONNX_OCR_MODEL_DIR` | `datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b`, including the five math assets |
| `ONNX_CLIP_MODEL_DIR` / `ONNX_CLIP_INT8_MODEL_DIR` | `Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523`, FP32 / INT8 |
| `ONNX_WHISPER_MODEL_DIR` / `ONNX_WHISPER_INT8_MODEL_DIR` | `onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7`, FP32 / INT8 |
| `ONNX_DEPTH_MODEL_DIR` / `ONNX_DEPTH_INT8_MODEL_DIR` | `onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e`, FP32 / INT8 |
| `ONNX_WHISPER_WAV` | Absolute path to `tests/fixtures/speech.wav` |
| `PDFIUM_LIBRARY` | Absolute path to a platform-compatible native Pdfium library |

```sh
cargo test --locked --offline --lib -- --include-ignored
```

The tests check inference output, beyond session construction:

- CLIP: 512-element embeddings, text label first above 0.2 cosine, matching
  precision rankings within 0.1; top-k clamping and cancellation.
- Whisper: complete normalized fixture output, English detection on repeated
  requests, request-local decoder state. Tiny substitutes `roots` and `note` for
  spoken `routes` and `node`; see [fixture expectations](../tests/fixtures/README.md).
- Depth: real graph plus external data, finite maps, FP32/INT8 correlation above
  0.9, valid previews including extreme and very small numeric ranges.
- Indra: exact Korean/English scan text, 35 lines and eight layout blocks.
- Math: all five additional graphs, classifier selection, complete squared
  equation, EOS truncation rejection and fresh decoding state.
- Pdfium: scanned OCR, text-only extraction without a session, DPI coordinates,
  missing/wrong/correct encrypted-PDF passwords, styled line grouping, nested
  form image bounds, hybrid text with columns preserved, and blank-image pages.

Original PDF/math fixtures and their generator are checked in under
[`tests/fixtures`](../tests/fixtures/README.md). The generator needs macOS;
consuming the fixtures needs only Pdfium and the listed weights.

## Live Mesh acceptance and hot loading

[`tests/managed_hot_load_live.rs`](../tests/managed_hot_load_live.rs) launches the
built `mesh-onnx` binary as a command plugin of a real Mesh host. Build the host
from the SDK-compatible Mesh revision with `just mesh`, then set:

```sh
export MESH_ONNX_TEST_HOST=/absolute/path/to/mesh-llm/target/debug/mesh-llm
# Retain this host-owned Hugging Face cache between test runs.
export MESH_ONNX_TEST_HF_CACHE=/absolute/path/to/acceptance-hf-cache
cargo test --locked --test managed_hot_load_live -- --include-ignored --nocapture
```

Also export every fixture variable above. The asset-service case needs internet
on its first run; it requests only the CLIP INT8 recipe's six files. Its second
node starts with `HF_HUB_OFFLINE=1` and must use those cached files. The other
cases use local bundles and perform no weight downloads.

The harness reserves independent ports and data directories and cleans up only
its own child processes. Inference goes through Mesh's HTTP ingress, including
all four families from a separate client-only Mesh node. The provider's loopback
`/v1/models` inventory is observed for residency and `assets_ready` transitions;
reading inventory must never load an engine.

Coverage includes all seven supported family/precision combinations, racing
first requests, cross-family concurrency, idle eviction with byte sizes and
modification times unchanged, loading again in the same provider process,
startup loading, disk-only prefetch, zero disabling idle eviction, shared failed
load causes, depth PNG, transcription plain text and body-format precedence,
400/406 responses, plus scanned/encrypted/hybrid PDF requests through Mesh. A
separate two-node case asserts that a client-only node discovers all four model
kinds with their workload classes, that discovery wakes no engine, and that each
kind is then callable from that node.

## Lifecycle alignment with Mesh #2704

The provider follows the lifecycle contract in
[Mesh #2704](https://github.com/Mesh-LLM/mesh-llm/pull/2704):

| Setting or transition | ONNX behavior |
|---|---|
| `load` omitted / `startup` | Prepare assets and create the session at initialization |
| `load = "on_demand"` | First named inference acquires the engine; queued callers share the load |
| Failed load | Queued callers receive the original cause; retries use a five-second backoff |
| `prefetch = true` (default) | Prepare disk assets without an engine for an on-demand model |
| `idle_timeout_secs = 0` (default) | Keep the loaded engine resident |
| Positive idle timeout | Drop an idle engine; queued/active leases prevent eviction; cached files remain |

ONNX engines live inside the provider process. Native GGUF instances in #2704
have separate backend processes, so their port/instance-ID assertions are native
specific. ONNX assertions observe inventory transitions and successful cold reload.
Text-mode PDF extraction intentionally bypasses engine residency.

## Recorded state and remaining limits

The macOS arm64 real-weight library suite passes 77 tests with no ignored tests.
All five live acceptance cases pass, including host-owned asset prefetch, fresh
offline loading, FP32 and INT8 through two Mesh nodes, and a client-only node
that discovers all four model kinds in `/api/models` and then calls each of
them across the mesh. The paired run uses a host built from the merged main
commit `71a28b5830d81bf83effcf1e08d37377540cf829` (#2704), which the SDK pin
names too.
The full-graph checks use `ort` 2.0.0-rc.12 and Rust 1.98.

A serve node advertises its plugin models in the mesh catalog slightly before it
claims the host role that makes them callable. Discovery can therefore name a
model a few seconds before a peer-issued request for it is routable; the live
harness waits for the serving peer rather than failing on that ordering, and the
client-side assertions below cover both states.

Follow-up regressions check that Text mode never rasterizes, capped/rotated
text geometry matches rendering, clipped image bounds do not duplicate an
authoritative text layer even when OCR spelling differs, and recursive image
traversal checks cancellation within a shared object budget. The live harness
also tests startup port-error classification and bounded retries, waiting for
its owned child's readiness event before publishing the selected URLs.

This run certifies CPU behavior on macOS arm64 and two processes on one machine.
It does not certify separate physical hosts, GPU providers or mathematical/OCR
accuracy beyond these fixtures. Text/hybrid PDF blocks have generic Text labels;
model-backed OCR supplies layout labels. Mesh's chat-only `/v1/models` can remain
empty on a node serving only these task endpoints.

The prepared GitHub Actions workflow could not be pushed because the available
OAuth token lacks `workflow` scope. The reviewable template is
[`ci/rust.yml.example`](../ci/rust.yml.example); publish it as
`.github/workflows/rust.yml` using an authorized identity. Local acceptance
tests are checked into this PR independently of workflow publication.
