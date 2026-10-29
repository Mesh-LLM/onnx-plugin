# ONNX OCR plugin for Mesh LLM

This plugin runs [Indra OCR](https://github.com/datalab-to/indra)'s document models inside a Rust process using [`ort`](https://ort.pyke.io/). It registers a `document_ocr_v1` inference endpoint with Mesh. Mesh discovers the ready model from `/v1/models`, routes `POST /v1/documents/ocr` to a serving node, and relays the multipart document to this provider.

## Build and install

```sh
cargo build --release
```

Download the Indra model assets from the [Indra OCR model repository](https://huggingface.co/datalab-to/indra-ocr) at revision `v1.1.0`. The directory needs `det.onnx`, `rec.onnx`, `order.onnx`, `script.onnx`, any available `rec_<group>.onnx` files, and `meta.json`. The native plugin never starts Python or an `onnxruntime` Python worker. Model weights remain outside the binary.

Configure the plugin with absolute paths:

```toml
[[plugin]]
name = "onnx-ocr"
command = "/absolute/path/to/mesh-onnx-ocr"
args = [
  "--model-dir", "/models/indra-ocr-v1.1.0",
  "--pdfium-library", "/absolute/path/to/libpdfium.so"
]
```

The `--pdfium-library` option enables PDF rasterization through Pdfium. Use the matching library file for your platform (`.so`, `.dylib`, or `.dll`); image input needs no Pdfium. Package that library alongside the plugin or point to a system installation. The `ort` build obtains ONNX Runtime for the target platform; inspect the resulting binary's library dependencies when packaging it.

## OCR request

```sh
curl -F model=datalab/indra-ocr -F file=@document.pdf \
  http://127.0.0.1:10000/v1/documents/ocr
```

Multipart fields are `model`, `file`, `response_format` (`json`, `text`, or `markdown`), and `max_pages` (`0` processes up to 1,000 pages; explicit limits must be between 1 and 1,000). Inputs are PDF or a supported image format. The maximum file size is 64 MiB. PDF pages render at 200 DPI, capped at 4,096 pixels per dimension, and are recognized one at a time. JSON returns the model ID and page indexed `text`, `markdown`, `lines` with boxes and confidence, and ordered layout `blocks`. Errors use `error.message`.

The native port currently runs Indra's orientation model, detector, script classifier, base and script specific recognizers, and reading order and layout models. It does not yet reproduce all upstream postprocessing. On the pinned scanned page, this implementation detects 35 lines and 8 layout blocks; upstream Indra reports 36 lines and 9 blocks. The functional tests pin the native results and key text so regressions are visible while that gap is closed. The [Mesh core OCR work](https://github.com/Mesh-LLM/mesh-llm) adds workload-aware discovery and routing for this endpoint.

## Functional tests

The fixture image and a PDF made from it are checked in under `tests/fixtures/`. The model weights and Pdfium library are external. Run the opt-in tests with the Indra `v1.1.0` assets and a platform Pdfium library:

```sh
ONNX_OCR_MODEL_DIR=/absolute/path/to/indra-ocr-v1.1.0 \
ONNX_OCR_PDFIUM_LIBRARY=/absolute/path/to/libpdfium.dylib \
  cargo test --lib -- --ignored
```

The tests execute the real ONNX detector, recognition, orientation, and layout sessions, render the PDF with Pdfium, and submit both fixtures to the plugin's HTTP endpoint. They check page structure and expected Korean and English text. On Linux or Windows, point `ONNX_OCR_PDFIUM_LIBRARY` to the corresponding library file.

Indra's code is Apache-2.0. Its [model weights use modified OpenRAIL-M terms](https://huggingface.co/datalab-to/indra-ocr/blob/main/LICENSE); check those terms before redistributing weights or hosting public inference.
