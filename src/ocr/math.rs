//! Optional Indra v1.1.0 math classifier and cached LaTeX line decoder.

use super::{
    Line,
    recognition::{crop_line, line_inputs},
    session_with_threads,
};
use anyhow::{Context, Result, bail};
use image::GrayImage;
use ort::{
    session::Session,
    value::{DynValue, Tensor},
};
use std::path::Path;

pub(super) struct MathReader {
    classifier: Session,
    encoder: Session,
    initial: Session,
    decoder: Session,
    height: usize,
    classifier_height: usize,
    max_tokens: usize,
    cache_shape: [usize; 5],
    tokens: Vec<String>,
}

struct Cache {
    self_k: DynValue,
    self_v: DynValue,
    cross_k: DynValue,
    cross_v: DynValue,
    mask: DynValue,
}

impl MathReader {
    pub fn load(directory: &Path, threads: usize, classifier_height: usize) -> Result<Self> {
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("math_meta.json"))
                .context("read Indra math_meta.json")?,
        )?;
        let config = &metadata["config"];
        let dimension = |key: &str, maximum: usize| -> Result<usize> {
            let value = config[key]
                .as_u64()
                .with_context(|| format!("Indra math {key} missing"))?;
            let value =
                usize::try_from(value).context("Indra math dimension exceeds platform size")?;
            if value == 0 || value > maximum {
                bail!("invalid Indra math {key}");
            }
            Ok(value)
        };
        let height = dimension("height", 1024)?;
        let max_tokens = dimension("max_tokens", 2048)?;
        let layers = dimension("dec_layers", 32)?;
        let heads = dimension("dec_heads", 32)?;
        let channels = dimension("channels", 4096)?;
        if !channels.is_multiple_of(heads) {
            bail!("invalid Indra math attention dimensions");
        }
        let mut tokens = vec![String::new(); 4];
        tokens.extend(
            metadata["tokens"]
                .as_array()
                .context("Indra math vocabulary missing")?
                .iter()
                .map(|token| {
                    token
                        .as_str()
                        .map(str::to_owned)
                        .context("invalid Indra math token")
                })
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(Self {
            classifier: session_with_threads(directory.join("mathcls.onnx"), threads)?,
            encoder: session_with_threads(directory.join("math_enc.onnx"), threads)?,
            initial: session_with_threads(directory.join("math_dec_init.onnx"), threads)?,
            decoder: session_with_threads(directory.join("math_dec_step.onnx"), threads)?,
            height,
            classifier_height,
            max_tokens,
            cache_shape: [layers, 1, heads, 0, channels / heads],
            tokens,
        })
    }

    pub fn recognize(&mut self, page: &GrayImage, lines: &mut [Line]) -> Result<()> {
        for line in lines {
            let crop = crop_line(page, line.bbox, self.classifier_height, 1)?;
            let (image, widths) = line_inputs(&crop)?;
            let outputs = self
                .classifier
                .run(ort::inputs!["x" => image, "widths" => widths])?;
            let (_, probabilities) = outputs[0].try_extract_tensor::<f32>()?;
            let is_math = probabilities
                .first()
                .is_some_and(|value| value.is_finite() && *value > 0.5);
            drop(outputs);
            if is_math {
                let crop = crop_line(page, line.bbox, self.height, 1)?;
                let (text, confidence) = self.read_line(&crop)?;
                if text.contains('$') && confidence >= 0.9 {
                    line.text = text;
                    line.confidence = confidence;
                }
            }
        }
        Ok(())
    }

    fn initial_cache(&mut self, crop: &GrayImage) -> Result<Cache> {
        let (image, widths) = line_inputs(crop)?;
        let mut encoded = self
            .encoder
            .run(ort::inputs!["x" => image, "widths" => widths])?
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        if encoded.len() != 2 {
            bail!("math encoder requires memory and mask outputs");
        }
        let mask = encoded.remove(1);
        let memory = encoded.remove(0);
        let mut inputs = ort::inputs!["memory" => memory.view()];
        if self
            .initial
            .inputs()
            .iter()
            .any(|input| input.name() == "mask")
        {
            inputs.push(("mask".into(), mask.view().into()));
        }
        let mut cross = self
            .initial
            .run(inputs)?
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        if cross.len() != 2 {
            bail!("math initial decoder requires cross_k and cross_v outputs");
        }
        Ok(Cache {
            cross_v: cross.remove(1),
            cross_k: cross.remove(0),
            mask,
            self_k: Tensor::from_array((self.cache_shape, Vec::<f32>::new()))?.into_dyn(),
            self_v: Tensor::from_array((self.cache_shape, Vec::<f32>::new()))?.into_dyn(),
        })
    }

    fn read_line(&mut self, crop: &GrayImage) -> Result<(String, f32)> {
        let mut cache = self.initial_cache(crop)?;
        let mut token = 1_i64;
        let mut sequence = Vec::new();
        let mut probabilities = Vec::new();
        for position in 0..self.max_tokens.saturating_sub(1) {
            let logits = self.decode_step(token, position, &mut cache)?;
            let (next, confidence) = most_likely(&logits)?;
            probabilities.push(confidence);
            if next == 2 {
                break;
            }
            if probabilities.len() >= 8 && mean(&probabilities) < 0.8 {
                break;
            }
            if next >= 4 {
                sequence.push(next);
            }
            token = i64::try_from(next).context("math token exceeds i64")?;
        }
        let sequence = sequence
            .into_iter()
            .map(|token| {
                self.tokens
                    .get(token)
                    .map(String::as_str)
                    .context("math token outside vocabulary")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((detokenize(&sequence), mean(&probabilities)))
    }

    fn decode_step(&mut self, token: i64, position: usize, cache: &mut Cache) -> Result<Vec<f32>> {
        let mut inputs = ort::inputs![
            "token" => Tensor::from_array(([1], vec![token]))?,
            "pos" => Tensor::from_array((Vec::<usize>::new(), vec![position as i64]))?,
            "self_k" => cache.self_k.view(), "self_v" => cache.self_v.view(),
            "cross_k" => cache.cross_k.view(), "cross_v" => cache.cross_v.view(),
        ];
        if self
            .decoder
            .inputs()
            .iter()
            .any(|input| input.name() == "mask")
        {
            inputs.push(("mask".into(), cache.mask.view().into()));
        }
        let mut outputs = self
            .decoder
            .run(inputs)?
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        if outputs.len() != 3 {
            bail!("math decoder requires logits and self_k/self_v outputs");
        }
        cache.self_v = outputs.remove(2);
        cache.self_k = outputs.remove(1);
        let (_, logits) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(logits.to_vec())
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn most_likely(logits: &[f32]) -> Result<(usize, f32)> {
    if logits
        .iter()
        .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        bail!("math decoder returned invalid logits");
    }
    let (index, &maximum) = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .context("math decoder returned no finite logits")?;
    let denominator = logits
        .iter()
        .map(|&value| f64::from(value - maximum).exp())
        .sum::<f64>();
    Ok((index, (1.0 / denominator) as f32))
}

fn detokenize(tokens: &[&str]) -> String {
    let mut output = String::new();
    let mut math = Vec::new();
    let mut inside = false;
    for &token in tokens {
        if token == "$" {
            if inside {
                output.push('$');
                output.push_str(&math.join(" "));
                output.push('$');
                math.clear();
            }
            inside = !inside;
        } else if inside {
            math.push(token);
        } else {
            output.push_str(token);
        }
    }
    if inside && !math.is_empty() {
        output.push('$');
        output.push_str(&math.join(" "));
        output.push('$');
    }
    output.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn latex_tokens_preserve_spacing_and_close_partial_math() {
        assert_eq!(
            detokenize(&["we ", "have ", "$", "x", "_", "i", "\\leq", "1", "$"]),
            "we have $x _ i \\leq 1$"
        );
        assert_eq!(detokenize(&["$", "\\frac", "1", "2"]), "$\\frac 1 2$");
    }
    #[test]
    fn decoder_probability_is_stable_and_invalid_outputs_fail() {
        let (token, probability) = most_likely(&[1000.0, 1001.0, f32::NEG_INFINITY]).unwrap();
        assert_eq!(token, 1);
        assert!((probability - 0.7310586).abs() < 0.00001);
        assert!(most_likely(&[f32::NAN]).is_err());
        assert!(most_likely(&[f32::NEG_INFINITY]).is_err());
    }
}
