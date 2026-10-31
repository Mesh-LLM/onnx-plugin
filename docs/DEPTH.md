# Depth Anything V2 Small

Source: [onnx-community/depth-anything-v2-small-ONNX](https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX/tree/c3b67641fd837b2368757101311e5d21e511441e).

## When to use it

- **Judging whether a photo is assessable.** Depth says what is near and what is far across the frame, which is enough to flag a shot taken across a room, where the detail is too small to assess, or one so close that the subject fills the frame. That is a documentation-quality check, not a measurement.
- **Foreground and background separation.** The map is a cheap way to decide what to crop for, or which part of a scene a reviewer should look at.
- **Scene layout hints.** Relative ordering of surfaces for downstream heuristics, without a second vision model.

In the [claim-intake pipeline](HTTP_EXAMPLES.md#worked-example-one-claim-submission-four-models) this is step three: the damage photo gets a usability flag before anyone is asked to review it.

**The units are the constraint.** Values are relative depth in the model's own scale, consistent within a single image and not comparable across images, so nothing here reports centimetres or distance to an object. The PNG preview is normalized per image for the same reason: two previews are not a measurement of each other. Anything that needs real distance needs a calibrated capture, not this model.

```toml
[[models]]
model = "onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e"
provider = "onnx"
variant = "int8"
[models.options]
threads = 4
[models.request_defaults]
output_resolution = "input"
visualization = "none"
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F 'model=onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e' \
  -F 'file=@scene.jpg' -F 'output_resolution=native' \
  http://localhost:9337/mesh/v1/images/depth
```

`Accept: application/json` returns `model`, `width`, `height`, `units: "relative"` and a finite row-major `values` array. This is relative scene depth, not meters. `output_resolution=input` resizes to the uploaded image grid; `native` retains the model output grid. Output is limited to four million pixels; source images are also bounded. Preprocessing follows the pinned processor's aspect-preserving resolution/multiple and normalization.

`visualization=normalized` adds a display-only `preview` containing `encoding: "grayscale_u8"`, dimensions and row-major `pixels`. This maps the map's minimum/maximum to 0/255 (constant maps to zero); it does not replace or change raw values. It is not a metric depth image or a cross-image calibration.

`Accept: image/png` returns that same preview as an 8-bit greyscale PNG instead of a JSON body, so the pixels never travel as a JSON array:

```sh
curl --fail-with-body -H 'Accept: image/png' \
  -F 'model=onnx-community/depth-anything-v2-small-ONNX@c3b67641fd837b2368757101311e5d21e511441e' \
  -F 'file=@scene.jpg' \
  -o depth.png http://localhost:9337/mesh/v1/images/depth
```

Asking for an image is itself the request for a preview, so this ignores the `visualization` default. The preview normalizes per image, so two PNGs are not comparable pixel for pixel; use the JSON body when the numeric range matters.

Every precision variant needs both the graph and its matching external-data file: `onnx/model.onnx` with `onnx/model.onnx_data` for `fp32`, and `onnx/model_quantized.onnx` with `onnx/model_quantized.onnx_data` for `int8`. The int8 export publishes 4-bit `MatMulNBits` weights under the `_quantized` name rather than the `_int8` convention the other exports follow, and an external-data graph names its weights file internally, so the pair cannot be renamed to a neutral name — a renamed copy downloads from a path that does not exist and would not load anyway. A graph-only directory is incomplete, even if the small graph file exists. The explicit recipe keeps relative paths intact. Graph compatibility, external-data loading and FP32/INT8 agreement are covered by the weight tests: both variants produce finite relative maps and correlate above 0.9, which is also the `MatMulNBits` contrib-ops check (see [validation](TESTING.md)).
