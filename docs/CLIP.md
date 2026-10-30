# CLIP ViT-B/32

Source: [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32/tree/6b7c523703ae26e39be4b03774966f8571c7f523).

## When to use it

- **Routing media you did not label.** Supply the candidate labels per request and get them ranked, so the taxonomy lives in the application and can change without touching the model or refetching weights. "Which queue does this photo belong to?" is the canonical call.
- **Triage before a human looks.** A cheap first pass that sorts inbound images into queues, with the score deciding whether a person sees it at all.
- **Zero-shot checks.** "Is this a receipt, a product photo or a screenshot?" — labels as questions, no training run.

In the [claim-intake pipeline](HTTP_EXAMPLES.md#worked-example-one-claim-submission-four-models) this is step two: the damage photo is ranked against damage categories the claims team defines.

**What it does not do.** It ranks against labels; there is no HTTP route that returns a bare image embedding, so vector search or "find similar images" needs an adapter change first. Scores are cosine similarities from the dual encoders, not calibrated probabilities, so use the ranking and the gaps between adjacent scores rather than reading a score as confidence.

```toml
[[models]]
model = "Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523"
provider = "onnx"
variant = "int8"
[models.options]
threads = 4
[models.request_defaults]
top_k = 3
truncate_text = false
[models.lifecycle]
load = "on_demand"
idle_timeout_secs = 300
prefetch = true
```

```sh
curl --fail-with-body -H 'Accept: application/json' \
  -F 'model=Xenova/clip-vit-base-patch32@6b7c523703ae26e39be4b03774966f8571c7f523' \
  -F 'file=@cat.png' -F 'labels=["a photo of a cat","a photo of a car"]' \
  -F 'top_k=2' http://localhost:9337/mesh/v1/images/similarity
```

Returns `model`, `score_type: "cosine_similarity"` and descending `scores` entries with `label` and `score`. These are normalized embedding cosine similarities, not probabilities or upstream softmax classification confidences. Do not compare their values as calibrated probabilities. The dual-encoder adapter does not invent a learned logit scale absent from the selected export.

Labels are a required JSON array in a multipart field, limited to 100 nonempty texts, each at most 4096 UTF-8 bytes. `top_k` is 1–100 and the result contains at most the number of labels supplied. Text exceeding the model's 77-position token context is rejected unless `truncate_text=true`; truncation preserves the end token. Image preprocessing follows the pinned CLIP crop and normalization. Projection size is 512 and comes from compatible metadata.

The Rust adapter also exposes separate image/text embedding methods. The HTTP task here is image-plus-label ranking; it does not overload the text-only OpenAI embeddings request with image uploads. Graph compatibility and ranking are covered by the weight tests: both encoders produce 512-dimension vectors, the scanned fixture ranks its text label first above 0.2, and INT8 agrees with FP32 label for label within 0.1 cosine (see [validation](TESTING.md)).
