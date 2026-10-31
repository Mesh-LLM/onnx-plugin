//! Indra line crop preparation, script routing, and CTC recognition.

use std::{collections::BTreeMap, path::Path};

use anyhow::{Context, Result, bail};
use image::{GrayImage, imageops::FilterType};
use ort::{inputs, session::Session, value::Tensor};
use unicode_normalization::UnicodeNormalization;

use super::{Line, session_with_threads};

const PAD_X: f32 = 0.3;
const PAD_Y: f32 = 0.2;
const MIN_CONF: f32 = 0.5;
const ROUTE_SURE: f32 = 0.95;

struct Vocab {
    chars: Vec<String>,
}

impl Vocab {
    fn from_meta(metadata: &serde_json::Value, key: &str) -> Result<Self> {
        let encoded = metadata[key]
            .as_str()
            .with_context(|| format!("Indra {key} missing"))?;
        let value: serde_json::Value = serde_json::from_str(encoded)?;
        if value["blank"].as_u64() != Some(0) {
            bail!("unsupported Indra vocabulary blank token");
        }
        let chars = value["chars"]
            .as_array()
            .context("Indra vocabulary characters missing")?
            .iter()
            .map(|entry| {
                entry
                    .as_str()
                    .map(str::to_owned)
                    .context("invalid vocabulary token")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { chars })
    }

    fn decode(&self, ids: &[i64]) -> String {
        ids.iter()
            .filter_map(|id| usize::try_from(*id).ok())
            .filter_map(|id| id.checked_sub(1))
            .filter_map(|index| self.chars.get(index))
            .map(String::as_str)
            .collect::<String>()
            .nfc()
            .collect()
    }
}

pub struct Recognizers {
    base: Session,
    vocab: Vocab,
    script: Option<Session>,
    script_groups: Vec<String>,
    groups: BTreeMap<String, (Session, Vocab)>,
    height: usize,
    window: usize,
}

impl Recognizers {
    pub fn load(model_dir: &Path, metadata: &serde_json::Value, threads: usize) -> Result<Self> {
        let base = session_with_threads(model_dir.join("rec.onnx"), threads)?;
        let vocab = Vocab::from_meta(metadata, "vocab")?;
        let script = if model_dir.join("script.onnx").is_file() {
            Some(session_with_threads(
                model_dir.join("script.onnx"),
                threads,
            )?)
        } else {
            None
        };
        let script_groups = metadata["script_groups"]
            .as_array()
            .map(|groups| {
                groups
                    .iter()
                    .filter_map(|name| name.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_else(|| ["base", "cjk", "rtl", "other"].map(str::to_owned).to_vec());
        let mut groups = BTreeMap::new();
        if let Some(map) = metadata.as_object() {
            for name in map.keys().filter_map(|key| key.strip_prefix("vocab_")) {
                let path = model_dir.join(format!("rec_{name}.onnx"));
                if path.is_file() {
                    groups.insert(
                        name.to_owned(),
                        (
                            session_with_threads(path, threads)?,
                            Vocab::from_meta(metadata, &format!("vocab_{name}"))?,
                        ),
                    );
                }
            }
        }
        Ok(Self {
            base,
            vocab,
            script,
            script_groups,
            groups,
            height: metadata["rec_height"]
                .as_u64()
                .context("Indra rec_height missing")? as usize,
            window: metadata["rec_window"].as_u64().unwrap_or(1) as usize,
        })
    }

    pub fn recognize_lines(&mut self, page: &GrayImage, boxes: &[[f32; 4]]) -> Result<Vec<Line>> {
        let mut lines = Vec::with_capacity(boxes.len());
        for bbox in boxes {
            let band = crop_line(page, *bbox, self.height, 1)?;
            let tall = if self.window == 1 {
                band.clone()
            } else {
                crop_line(page, *bbox, self.height, self.window)?
            };
            let (group, sure) = self.classify(&band)?;
            let mut best = if group == "base" || !sure {
                self.read_base(&tall)?
            } else {
                (String::new(), -1.0)
            };
            if group != "base" {
                if let Some((session, vocab)) = self.groups.get_mut(&group) {
                    let candidate = read_line(session, &band, vocab)?;
                    if candidate.1 > best.1 {
                        best = candidate;
                    }
                } else if best.1 < 0.0 {
                    best = self.read_base(&tall)?;
                }
            }
            lines.push(Line {
                text: best.0,
                bbox: *bbox,
                confidence: best.1.max(0.0),
            });
        }
        Ok(lines)
    }

    fn read_base(&mut self, crop: &GrayImage) -> Result<(String, f32)> {
        read_line(&mut self.base, crop, &self.vocab)
    }

    fn classify(&mut self, crop: &GrayImage) -> Result<(String, bool)> {
        let Some(session) = self.script.as_mut() else {
            return Ok(("base".into(), true));
        };
        let (image, widths) = line_inputs(crop)?;
        let outputs = session.run(inputs!["x" => image, "widths" => widths])?;
        let (shape, probabilities) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() == 1 {
            let p = probabilities[0];
            let group = if p > 0.5 { "cjk" } else { "base" };
            return Ok((group.into(), p.max(1.0 - p) > ROUTE_SURE));
        }
        let (index, probability) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .context("script classifier returned no classes")?;
        let group = self
            .script_groups
            .get(index)
            .cloned()
            .unwrap_or_else(|| "base".into());
        Ok((group, *probability > ROUTE_SURE))
    }
}

fn read_line(session: &mut Session, crop: &GrayImage, vocab: &Vocab) -> Result<(String, f32)> {
    let (image, widths) = line_inputs(crop)?;
    let outputs = session.run(inputs!["x" => image, "widths" => widths])?;
    let (id_shape, ids) = outputs["ids"].try_extract_tensor::<i64>()?;
    let (_, confidences) = outputs["conf"].try_extract_tensor::<f32>()?;
    let (_, lengths) = outputs["lengths"].try_extract_tensor::<i64>()?;
    let steps = usize::try_from(lengths[0])
        .context("invalid CTC length")?
        .min(usize::try_from(id_shape[1])?);
    let mut kept = Vec::new();
    let mut confidence_sum = 0.0;
    let mut previous = 0;
    for index in 0..steps {
        let id = ids[index];
        if id != 0 && id != previous {
            kept.push(id);
            confidence_sum += confidences[index];
        }
        previous = id;
    }
    let confidence = if kept.is_empty() {
        0.0
    } else {
        confidence_sum / kept.len() as f32
    };
    let text = if confidence >= MIN_CONF {
        vocab.decode(&kept)
    } else {
        String::new()
    };
    Ok((text, confidence))
}

pub(super) fn line_inputs(crop: &GrayImage) -> Result<(Tensor<f32>, Tensor<i64>)> {
    let width = crop.width() as usize;
    let height = crop.height() as usize;
    let padded_width = width.div_ceil(8) * 8;
    let mut data = vec![0.0_f32; height * padded_width];
    for y in 0..height {
        for x in 0..width {
            data[y * padded_width + x] = 1.0 - crop.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
    let image = Tensor::from_array(([1, 1, height, padded_width], data))?;
    let widths = Tensor::from_array(([1], vec![width as i64]))?;
    Ok((image, widths))
}

pub(super) fn crop_line(
    page: &GrayImage,
    bbox: [f32; 4],
    rec_height: usize,
    rec_window: usize,
) -> Result<GrayImage> {
    let line_height = (bbox[3] - bbox[1]).max(1.0);
    let pad_y = (rec_window as f32 * (1.0 + 2.0 * PAD_Y) - 1.0) / 2.0;
    let left = (bbox[0] - PAD_X * line_height).floor() as i32;
    let right = (bbox[2] + PAD_X * line_height).ceil() as i32;
    let top = (bbox[1] - pad_y * line_height).floor() as i32;
    let bottom = (bbox[3] + pad_y * line_height).ceil() as i32;
    let crop_width = (right - left).max(1) as usize;
    let crop_height = (bottom - top).max(1) as usize;
    let mut pixels = vec![255_u8; crop_width * crop_height];
    for y in 0..crop_height {
        let source_y = top + y as i32;
        if source_y < 0 || source_y >= page.height() as i32 {
            continue;
        }
        for x in 0..crop_width {
            let source_x = left + x as i32;
            if source_x >= 0 && source_x < page.width() as i32 {
                pixels[y * crop_width + x] = page.get_pixel(source_x as u32, source_y as u32)[0];
            }
        }
    }
    let crop = GrayImage::from_raw(crop_width as u32, crop_height as u32, pixels)
        .context("construct line crop")?;
    let target_height = rec_height * rec_window;
    let target_width = ((crop_width as f32 * target_height as f32 / crop_height as f32).round()
        as usize)
        .clamp(8, 4096);
    Ok(image::imageops::resize(
        &crop,
        target_width as u32,
        target_height as u32,
        FilterType::Triangle,
    ))
}
