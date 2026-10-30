# Whisper Tiny

Source: [onnx-community/whisper-tiny](https://huggingface.co/onnx-community/whisper-tiny/tree/ff4177021cc41f7db950912b73ea4fdf7d01d8e7).

## When to use it

- **Voice notes into searchable text.** A technician, driver or inspector records a note instead of typing; the transcript joins the record and can be searched.
- **Short call or meeting snippets.** Anything under 30 seconds, where the value is the words rather than who spoke when.
- **Language-agnostic intake.** `language = "auto"` detects the language, so a mixed-language queue needs no routing decision up front; the response reports what it chose.

In the [claim-intake pipeline](HTTP_EXAMPLES.md#worked-example-one-claim-submission-four-models) this is the last step: the technician's own account alongside the structured fields.

**The window is the design constraint.** The adapter does one 30-second pass and rejects longer audio rather than truncating it, so a client with longer recordings must split them and join the text, accepting that a sentence spanning a cut may be garbled. Timestamps, translation and streaming are not offered, and no fallback silently produces them.

```toml
[[models]]
model = "onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7"
provider = "onnx"
variant = "int8"
[models.options]
threads = 4
[models.request_defaults]
language = "auto"
max_output_tokens = 128
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F 'model=onnx-community/whisper-tiny@ff4177021cc41f7db950912b73ea4fdf7d01d8e7' \
  -F 'file=@speech.wav' -F 'language=en' \
  http://localhost:9337/v1/audio/transcriptions
```

Returns `model`, `text`, selected `language`, and `generated_tokens`. `Accept: text/plain` returns only transcript text.

Because this is the OpenAI-shaped route, `response_format` is accepted as the form field OpenAI clients send: `json` (default) or `text`. `verbose_json`, `srt` and `vtt` are refused with a reason rather than answered with a body that lacks the timings they promise — this decoder produces no timestamps. When a caller sends both a header and the field, the field wins, because naming the body in the form is that API's own mechanism.

The initial adapter accepts bounded 16 kHz PCM16 or float WAV, mixing supported stereo input to mono, up to 30 seconds. Other encodings, sample rates, long-audio processing, translation, timestamps and realtime streaming are not silently approximated. Convert audio before upload if necessary. Model processor metadata fixes 80 Mel features and the audio context; these are not user knobs.

`language` is `auto` or a language token supported by the pinned tokenizer. `max_output_tokens` is 1–444; termination also respects the model context, prompt length and end token. Decoding is greedy. Each request owns its decoder KV state; cancellation is checked between steps. Engines can be shared without sharing user audio state.

This plugin serves the existing Mesh transcription route. Built-in Whisper can remain another provider of the same task. Graph compatibility and transcript quality are covered by the weight-dependent tests: the spoken fixture transcribes to its expected phrases with auto-detected `language = en`, and a second pass reproduces the transcript exactly (see [validation](TESTING.md)).

WAVE_FORMAT_EXTENSIBLE files are accepted for PCM16 and IEEE float32 subformats. Finite float samples outside `[-1, 1]` are clipped before channel mixing; NaN and infinities are rejected. The pinned Tiny model makes two word substitutions on the speech fixture (`roots`/`note` for spoken `routes`/`node`); the complete expected output is recorded in [the fixture guide](../tests/fixtures/README.md).
