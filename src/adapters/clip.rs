//! CLIP dual-encoder inference and cosine similarity ranking.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::DynamicImage;
use ort::{session::Session, value::Tensor};
use serde::Serialize;
use tokenizers::Tokenizer;

use super::{Variant, graph, pixels, read_json, session, vector_output};

const IMAGE_SIZE: u32 = 224;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// A label and cosine similarity, sorted highest first. Scores are not probabilities.
#[derive(Debug, Clone, Serialize)]
pub struct LabelScore {
    pub label: String,
    pub score: f32,
}

pub struct Clip {
    vision: Session,
    text: Session,
    tokenizer: Tokenizer,
    text_limit: usize,
    projection_dim: usize,
    eos_token: u32,
}

impl Clip {
    pub fn load(directory: &Path, threads: usize, variant: Variant) -> Result<Self> {
        let config = read_json(directory, "config.json")?;
        let text_limit = config["text_config"]["max_position_embeddings"]
            .as_u64()
            .context("CLIP text context missing")? as usize;
        let projection_dim = config["projection_dim"]
            .as_u64()
            .context("CLIP projection dimension missing")? as usize;
        if text_limit != 77 || projection_dim != 512 {
            bail!("unsupported CLIP recipe: expected ViT-B/32 with 77 tokens and 512 dimensions");
        }
        let mut tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("load CLIP tokenizer: {error}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow::anyhow!("configure CLIP tokenizer: {error}"))?;
        let eos_token = tokenizer
            .token_to_id("<|endoftext|>")
            .context("CLIP end-of-text token missing")?;
        Ok(Self {
            vision: session(&graph(directory, "vision_model", variant), threads)?,
            text: session(&graph(directory, "text_model", variant), threads)?,
            tokenizer,
            text_limit,
            projection_dim,
            eos_token,
        })
    }

    pub fn image_embedding(&mut self, image: &DynamicImage) -> Result<Vec<f32>> {
        let cropped = pixels::center_crop(image, IMAGE_SIZE)?;
        let tensor = Tensor::from_array((
            [1, 3, IMAGE_SIZE as usize, IMAGE_SIZE as usize],
            pixels::normalized_chw(&cropped, MEAN, STD),
        ))?;
        let outputs = self.vision.run(ort::inputs!["pixel_values" => tensor])?;
        normalize(vector_output(
            &outputs,
            "image_embeds",
            self.projection_dim,
        )?)
    }

    pub fn text_embedding(&mut self, text: &str, truncate: bool) -> Result<Vec<f32>> {
        if text.len() > 64 * 1024 {
            bail!("text exceeds 65536 UTF-8 bytes");
        }
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("tokenize CLIP text: {error}"))?;
        let (ids, mask) = padded_tokens(
            encoding.get_ids(),
            self.text_limit,
            self.eos_token,
            truncate,
        )?;
        let ids = Tensor::from_array(([1, self.text_limit], ids))?;
        let mut inputs = ort::inputs!["input_ids" => ids];
        if self
            .text
            .inputs()
            .iter()
            .any(|input| input.name() == "attention_mask")
        {
            inputs.push((
                "attention_mask".into(),
                Tensor::from_array(([1, self.text_limit], mask))?.into(),
            ));
        }
        let outputs = self.text.run(inputs)?;
        normalize(vector_output(&outputs, "text_embeds", self.projection_dim)?)
    }

    pub fn rank(
        &mut self,
        image: &DynamicImage,
        labels: &[String],
        top_k: usize,
        truncate: bool,
    ) -> Result<Vec<LabelScore>> {
        if labels.is_empty() || labels.len() > 256 {
            bail!("provide between 1 and 256 labels");
        }
        if top_k == 0 || top_k > labels.len() {
            bail!("top_k must be between 1 and the number of labels");
        }
        let image = self.image_embedding(image)?;
        let mut scores = labels
            .iter()
            .map(|label| {
                let text = self.text_embedding(label, truncate)?;
                Ok(LabelScore {
                    label: label.clone(),
                    score: cosine(&image, &text)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        scores.sort_by(|left, right| right.score.total_cmp(&left.score));
        scores.truncate(top_k);
        Ok(scores)
    }
}

fn padded_tokens(
    tokens: &[u32],
    limit: usize,
    eos: u32,
    truncate: bool,
) -> Result<(Vec<i64>, Vec<i64>)> {
    if tokens.is_empty() || limit < 2 {
        bail!("invalid CLIP token sequence");
    }
    if tokens.len() > limit && !truncate {
        bail!("text exceeds CLIP's {limit}-token context; enable truncate_text explicitly");
    }
    let count = tokens.len().min(limit);
    let mut ids = vec![i64::from(eos); limit];
    let mut mask = vec![0_i64; limit];
    for (index, token) in tokens.iter().take(count).enumerate() {
        ids[index] = i64::from(*token);
        mask[index] = 1;
    }
    if tokens.len() > limit {
        ids[limit - 1] = i64::from(eos);
    }
    Ok((ids, mask))
}

fn normalize(mut values: Vec<f32>) -> Result<Vec<f32>> {
    let norm = values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        bail!("model returned an invalid zero/nonfinite embedding");
    }
    for value in &mut values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(values)
}

fn cosine(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        bail!("embedding dimensions do not match");
    }
    let score = left.iter().zip(right).map(|(a, b)| a * b).sum::<f32>();
    if !score.is_finite() {
        bail!("nonfinite similarity score");
    }
    Ok(score.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_limit_is_explicit_and_truncation_preserves_eos() {
        assert!(padded_tokens(&[1, 2, 3, 4], 3, 9, false).is_err());
        assert_eq!(
            padded_tokens(&[1, 2, 3, 4], 3, 9, true).unwrap(),
            (vec![1, 2, 9], vec![1, 1, 1])
        );
        assert_eq!(
            padded_tokens(&[1, 9], 4, 9, false).unwrap(),
            (vec![1, 9, 9, 9], vec![1, 1, 0, 0])
        );
    }

    #[test]
    fn cosine_is_normalized_and_rejects_invalid_vectors() {
        let normalized = normalize(vec![3.0, 4.0]).unwrap();
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((cosine(&normalized, &normalized).unwrap() - 1.0).abs() < 1e-6);
        assert!(normalize(vec![0.0, 0.0]).is_err());
        assert!(normalize(vec![f32::NAN]).is_err());
    }

    #[test]
    #[ignore = "requires a pinned CLIP bundle; set ONNX_CLIP_MODEL_DIR and/or ONNX_CLIP_INT8_MODEL_DIR"]
    fn clip_runs_native_dual_encoders() {
        let labels = [
            "a scanned page of printed text".to_string(),
            "a photograph of a dog".to_string(),
            "a bar chart".to_string(),
        ];
        let image = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/scan.png"
        ))
        .unwrap();
        let mut rankings = Vec::new();
        for (variable, variant) in [
            ("ONNX_CLIP_MODEL_DIR", Variant::Fp32),
            ("ONNX_CLIP_INT8_MODEL_DIR", Variant::Int8),
        ] {
            let Ok(dir) = std::env::var(variable) else {
                continue;
            };
            let mut engine = Clip::load(Path::new(&dir), 1, variant).unwrap();
            let vision = engine.image_embedding(&image).unwrap();
            let text = engine
                .text_embedding("a page of printed text", false)
                .unwrap();
            assert_eq!(vision.len(), 512);
            assert_eq!(text.len(), 512);
            assert!(cosine(&vision, &text).unwrap().is_finite());
            let ranked = engine.rank(&image, &labels, labels.len(), false).unwrap();
            // The fixture is a photographed letter, so the text label must win
            // over the photo-of-a-dog and chart labels.
            assert_eq!(ranked[0].label, labels[0], "{variable} ranked {ranked:?}");
            rankings.push((variable, ranked));
        }
        assert!(
            !rankings.is_empty(),
            "set ONNX_CLIP_MODEL_DIR and/or ONNX_CLIP_INT8_MODEL_DIR"
        );
        // Quantization buys memory and latency; it must not buy different
        // answers. Both precisions rank the same label first, with close scores.
        if let [(_, fp32), (_, int8)] = rankings.as_slice() {
            for (full, quantized) in fp32.iter().zip(int8) {
                assert_eq!(full.label, quantized.label);
                assert!(
                    (full.score - quantized.score).abs() < 0.1,
                    "{}: fp32 {} vs int8 {}",
                    full.label,
                    full.score,
                    quantized.score
                );
            }
        }
    }
}
