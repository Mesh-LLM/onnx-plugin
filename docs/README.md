# ONNX model guides

Mesh selects models in `~/.mesh-llm/config.toml`, supplies their configuration to the `onnx` plugin and downloads explicit asset bundles through its shared Hugging Face cache. The plugin executes native ONNX Runtime sessions. Model-specific preprocessing and decoding are implemented in Rust.

| Model | Guide | HTTP task |
|---|---|---|
| Indra OCR v1.1.0 | [Indra](INDRA.md) | `/mesh/v1/documents/ocr` |
| CLIP ViT-B/32 | [CLIP](CLIP.md) | `/mesh/v1/images/similarity` |
| Whisper Tiny | [Whisper](WHISPER.md) | `/v1/audio/transcriptions` |
| Depth Anything V2 Small | [Depth](DEPTH.md) | `/mesh/v1/images/depth` |

Each guide starts with the use cases that task is actually good for and the constraint that bounds it. The [end-to-end example](HTTP_EXAMPLES.md#worked-example-one-claim-submission-four-models) runs all four on a single submission, and [the request reference](HTTP_EXAMPLES.md) covers every endpoint's fields in one place.

These changes require the accompanying Mesh managed-model implementation. The ordinary plugin package installer does not download model weights. See [configuration and lifecycle](CONFIGURATION.md) and [validation](TESTING.md).

The examples call Mesh's public inference listener on port 9337, not the plugin's private loopback port. Bindings and model IDs are identical for local and remote requests. Configure the model once on the serving node; a connected mesh client can invoke its task by ID.

## Implementation validation status

All four pinned graphs have been downloaded at their recorded revisions and run on real weights, in FP32 and INT8 where the recipe offers both. The 56-test weight suite pins a per-model numerical result — line-accurate OCR text, an exact transcript, a ranked label, correlated depth maps — rather than only constructing sessions, and one single-node pass through Mesh's inference listener is recorded with timings and payloads. See [validation](TESTING.md) for what each test asserts, and each guide for native adapter limits. Not yet covered: Indra math, encrypted PDFs, the hybrid page path, concurrent callers and a second node routing to a cold provider.

## Where this can go next

Nothing below is implemented. It is here so a future task is chosen against a real use case instead of by model availability, and so the path convention is decided once.

Path rule, already in force for the four shipped tasks: an endpoint that has an OpenAI shape goes under `/v1/` (`/v1/embeddings`, `/v1/rerank`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/decisions`). A task with no OpenAI equivalent goes under `/mesh/v1/<noun>/<verb>`. Mesh matches the request path to a workload class and forwards that same literal string to the plugin, so a new task needs a path, a protocol label, and arms in Mesh's route vocabulary — renaming one side alone is a runtime 404.

| Task | Path that would carry it | Representative ONNX export | Use case |
|---|---|---|---|
| Text embeddings | `/v1/embeddings` | `BAAI/bge-small-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2` | Retrieval over a document set: chunk, embed, then answer with passages instead of a whole folder in the prompt. The obvious first addition, because the endpoint and workload class already exist. |
| Reranking | `/v1/rerank` | `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2` | Retrieve 50 candidates cheaply, rerank to the 10 that reach the prompt. Turns a fuzzy embedding hit list into an ordered, defensible answer set. |
| Image embeddings | `/mesh/v1/images/embeddings` | the CLIP image tower already loaded | "Find photos like this one": duplicate claim photos, the same part photographed twice, a product catalogue that dedupes itself. Needs no new weights, only a route and an index. |
| Speech synthesis | `/v1/audio/speech` | Piper, Kokoro, VITS exports | Reading a generated answer aloud, voice notices for accessibility, IVR prompts without a cloud round trip. The mirror image of transcription and a task Mesh already labels. |
| Voice activity / speaker split | `/mesh/v1/audio/segments` | Silero VAD | Cut an hour-long call into utterances before Whisper sees it, and skip silence instead of transcribing it. Cheap, and it makes long audio usable at all. |
| Object detection | `/mesh/v1/images/detections` | YOLO-family, DETR exports | Count items on a pallet, find the table region before OCR, locate faces to redact. Boxes turn every later task into a crop instead of a whole page. |
| Image classification | `/mesh/v1/images/classification` | timm ViT/ConvNeXt exports | Screen an upload as defect or acceptable, or as safe or abuse, before it costs a human's attention. |
| Image captioning / VQA | `/mesh/v1/images/caption` | Florence-2, BLIP, Moondream exports | Alt text and auto-metadata for an asset library, or "does this photo show the damage?" answered against the claim. |
| Translation | `/mesh/v1/text/translation` | NLLB-200-distilled, `Helsinki-NLP/opus-mt-*` | Support tickets and contracts in a language nobody on the team reads. Pair with OCR for scanned foreign paperwork. |
| Token classification | `/mesh/v1/text/entities` | BERT NER, PII-redaction exports | Redact names, addresses and account numbers before a document is stored or logged, and extract fields without a template per form. |
| Normals / point clouds | `/mesh/v1/images/normals` | Marigold normals, Depth Anything variants | Measure a real object from photos rather than estimating relative depth, and feed 3D or AR inspection. |

The cheapest useful step is the first row: text embeddings reuse an endpoint, a workload class and a response shape that already exist, and they are the half of retrieval the shipped four cannot do.
