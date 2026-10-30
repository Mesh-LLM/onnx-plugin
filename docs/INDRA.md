# Indra OCR

Source: [datalab-to/indra-ocr, v1.1.0](https://huggingface.co/datalab-to/indra-ocr/tree/v1.1.0). The revision is intentional: newer model metadata or graphs may require a different adapter.

## When to use it

- **Turning paperwork into records.** Invoices, receipts, claim forms and service sheets arrive as PDFs or as phone photos of paper; this returns page text, Markdown and per-line boxes with confidence, which is what field extraction needs.
- **Skipping OCR you do not need.** `mode = "auto"` uses the PDF's text layer when it is usable and only recognizes pixels when it is not, so a born-digital form costs no OCR at all. The response's `source` says which path a page took, which is worth logging.
- **Pure text extraction.** `mode = "text"` reads the text layer without the automatic quality checks, for documents you know are clean.
- **Layout-aware conversion.** `blocks` carry a label and box per region, so a page can be split into title, body and table areas before Markdown is generated. `keep_furniture` decides whether headers and footers survive into that Markdown.

In the [claim-intake pipeline](HTTP_EXAMPLES.md#worked-example-one-claim-submission-four-models) this is step one: the scanned service form.

Text-only PDFs need no model session, so a `mode = "text"` request is answered without loading weights or entering the residency queue.

```toml
[[models]]
model = "datalab-to/indra-ocr@v1.1.0"
provider = "onnx"
variant = "fp32"
[models.options]
threads = 4
math = true
[models.request_defaults]
dpi = 200
mode = "auto"
image_ocr = true
keep_furniture = false
max_pages = 100
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

Image example, requiring no Pdfium:

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F 'model=datalab-to/indra-ocr@v1.1.0' -F 'file=@scan.png' \
  http://localhost:9337/mesh/v1/documents/ocr
```

JSON contains `model`, and `pages` containing `page_index`, `text`, `markdown`, `source`, `rotation`, `vertical`, `lines` (text, pixel bounding box, confidence), and layout `blocks`. For example, the response shape is:

```json
{"model":"datalab-to/indra-ocr@v1.1.0","pages":[{"page_index":0,"text":"Example text","markdown":"Example text","source":"ocr","rotation":0,"vertical":false,"lines":[{"text":"Example text","bbox":[10,20,150,40],"confidence":0.95}],"blocks":[]}]}
```

This is an illustrative schema example, not measured inference output. Text and Markdown representations join pages with a blank line:

```sh
curl --fail-with-body -H 'Accept: text/markdown' \
  -F 'model=datalab-to/indra-ocr@v1.1.0' -F 'file=@paper.pdf' \
  -F 'max_pages=10' -F 'dpi=200' -F 'mode=ocr' \
  http://localhost:9337/mesh/v1/documents/ocr
```

For PDF input, install a platform-compatible Pdfium library and set `ONNX_PDFIUM_LIBRARY` in the serving process environment or pass plugin args `['--pdfium-library', '/absolute/path/to/libpdfium.dylib']` (use TOML double quotes in actual configuration). Linux uses `.so`, Windows `.dll`. PDF rendering is bounded to 4096 pixels per dimension and DPI 36–600. `max_pages` is 1–1000; zero is rejected. `password` is a request-only multipart field for encrypted PDFs.

`mode=auto` uses a usable text layer and falls back to OCR. `mode=ocr` always recognizes pixels; `mode=text` uses PDF text without auto-quality checks and is rejected for raster images. `image_ocr=true` adds OCR for image regions on text-layer pages. `keep_furniture` affects Markdown headers/footers while retaining JSON structure. Native auto-quality checks are heuristic and are not claimed to match upstream exactly.

`math=true` selects the math classifier and encoder/decoder graph bundle as well as OCR assets. It uses request-local decoding state and confidence gating. Math is not yet covered by an automated weight test: it pulls five further graphs and a LaTeX decoder, so treat it as unverified until that test exists. Internal vocabulary, recognition height/window and script-group definitions come from pinned metadata, not user tuning.
