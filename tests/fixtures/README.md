# Functional test fixtures

`scan.png` is copied from `examples/samples/scan.png` in [datalab-to/indra](https://github.com/datalab-to/indra) at commit `e20355319b5aa5705a876a8d2b40aca606d7df5f`, under that project's Apache-2.0 license. `scan.pdf` was generated from the same image and exercises the PDF rendering path. The fixtures contain a scanned Korean and English letter; no model weights are included here.

`speech.wav` is generated, not sampled: 16 kHz mono PCM16 from the macOS `say`
voice, containing "The quick brown fox jumps over the lazy dog. Mesh routes
speech recognition to whichever node is serving the model." It is the Whisper
input for the real-model test. The complete decoded transcript is pinned below;
it contains recognition errors and is not claimed to reproduce the spoken words
verbatim. Regenerate it with:

```sh
say -v Samantha -o /tmp/whisper-fixture.aiff "The quick brown fox jumps over the lazy dog. Mesh routes speech recognition to whichever node is serving the model."
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/whisper-fixture.aiff tests/fixtures/speech.wav
```

The pinned `onnx-community/whisper-tiny` revision
`ff4177021cc41f7db950912b73ea4fdf7d01d8e7` returns this complete normalized
transcript in **both FP32 and INT8** for the checked-in WAV:

> the quick brown fox jumps over the lazy dog mesh roots speech recognition to whichever note is serving the model

The model recognizes spoken “routes” as “roots” and “node” as “note” in this
fixture. Tests assert that known output, English detection on both repeated
requests, and identical repeated transcripts to catch decoder-cache leakage.
Normalization splits on non-alphanumeric characters, drops empty tokens,
lowercases each word, and joins with single spaces. This ignores punctuation and
capitalization while retaining every decoded word. A changed voice or macOS
speech-engine version may require a separately reviewed fixture expectation.

`math.png` and the generated PDF fixtures are original test material created by
`generate_pdf_fixtures.swift`. The equation is Times-rendered
`x² + y² = z²`; the real math test checks classifier selection, complete LaTeX
decoding, safe handling of a truncated decode, and repeat-request isolation.
