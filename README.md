# ONNX inference plugin for Mesh

Run Indra OCR, CLIP ViT-B/32 image/text similarity, Whisper Tiny transcription and Depth Anything V2 Small using native Rust and ONNX Runtime. Mesh owns model configuration and asset acquisition; the plugin owns preprocessing and inference, with independent on-demand loading and idle eviction.

Start with the [model documentation](docs/README.md), including a configuration example and HTTP request for every supported model:

- [Indra OCR](docs/INDRA.md)
- [CLIP image/text similarity](docs/CLIP.md)
- [Whisper transcription](docs/WHISPER.md)
- [Depth estimation](docs/DEPTH.md)
- [End-to-end HTTP example](docs/HTTP_EXAMPLES.md#worked-example-one-claim-submission-four-models): one submission through all four models

For example, with the plugin installed and the companion Mesh managed-model change:

```toml
[[plugin]]
name = "onnx"
command = "/absolute/path/to/mesh-onnx"

[[models]]
model = "datalab-to/indra-ocr@v1.1.0"
provider = "onnx"
[models.options]
threads = 4
math = true
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

Run `mesh-llm serve`, then invoke OCR through Mesh's inference listener:

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F 'model=datalab-to/indra-ocr@v1.1.0' -F 'file=@scan.png' \
  http://localhost:9337/mesh/v1/documents/ocr
```

Mesh downloads the selected compatible asset bundle into its shared Hugging Face cache. With this policy, the plugin creates sessions on first use and drops them after five idle minutes; later requests reuse cached weights. Images need no Pdfium. PDF uploads require a platform Pdfium library: see the [Indra guide](docs/INDRA.md).

See [configuration](docs/CONFIGURATION.md) for local directories, source overrides, variants, request defaults and lifecycle semantics. Model-specific options are validated before downloads; one task does not inherit another task's controls.

## Build and test

```sh
cargo build --release
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
```

This implementation requires the companion Mesh model-asset RPC and provider configuration support. Model weights remain outside the binary. The `ort` crate supplies ONNX Runtime; inspect native library dependencies when packaging. Pdfium is optional for images.

Weights were unavailable during development, so the offline suite ran without them. Every pinned graph has since been downloaded at its recorded revision and exercised on real weights, FP32 and INT8 where the recipe offers both: 56 tests pass with `--include-ignored`, including per-model numerical pins rather than session-construction checks. See [validation](docs/TESTING.md) for what each weight test asserts and the one recorded single-node Mesh pass. Adapter limitations and response semantics are documented per model.

Indra code is Apache-2.0; its model weights use [modified OpenRAIL-M terms](https://huggingface.co/datalab-to/indra-ocr/blob/v1.1.0/LICENSE). Other model licenses and artifact provenance must be checked at their pinned source revisions before packaging or redistribution.
