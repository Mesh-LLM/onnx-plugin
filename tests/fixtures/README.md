# Functional test fixtures

`scan.png` is copied from `examples/samples/scan.png` in [datalab-to/indra](https://github.com/datalab-to/indra) at commit `e20355319b5aa5705a876a8d2b40aca606d7df5f`, under that project's Apache-2.0 license. `scan.pdf` was generated from the same image and exercises the PDF rendering path. The fixtures contain a scanned Korean and English letter; no model weights are included here.

`speech.wav` is generated, not sampled: 16 kHz mono PCM16 from the macOS `say`
voice, containing "The quick brown fox jumps over the lazy dog. Mesh routes
speech recognition to whichever node is serving the model." It is the Whisper
input for the real-model test, chosen so the transcript can be asserted word for
word instead of only checked for non-emptiness. Regenerate it with:

```sh
say -v Samantha -o /tmp/whisper-fixture.aiff "The quick brown fox jumps over the lazy dog. Mesh routes speech recognition to whichever node is serving the model."
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/whisper-fixture.aiff tests/fixtures/speech.wav
```
