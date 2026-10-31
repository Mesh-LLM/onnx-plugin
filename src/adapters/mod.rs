//! Native ONNX task adapters. Asset acquisition and residency belong to the host.

pub mod clip;
pub mod depth;
mod pixels;
pub mod whisper;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, bail};
use image::DynamicImage;
use ort::session::Session;
use serde::{Deserialize, Serialize};

pub use clip::LabelScore;
pub use depth::{DepthMap, OutputResolution};
pub use whisper::Transcription;

#[derive(Debug)]
pub struct InvalidRequest(pub &'static str);
impl std::fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for InvalidRequest {}

/// Native task implementation selected by a compatible model recipe.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Clip,
    Whisper,
    Depth,
}

/// Pre-exported precision variant. The adapter never quantizes at startup.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    #[default]
    Fp32,
    Int8,
}

impl Variant {
    fn filename(self, stem: &str) -> String {
        match self {
            Self::Fp32 => format!("{stem}.onnx"),
            Self::Int8 => format!("{stem}_int8.onnx"),
        }
    }
}

/// One loaded task instance. Drop releases its sessions and model-owned buffers.
pub enum Engine {
    Clip(Box<clip::Clip>),
    Whisper(Box<whisper::Whisper>),
    Depth(Box<depth::Depth>),
}

impl Engine {
    /// Load a complete, already acquired model bundle from its local directory.
    pub fn load(
        kind: ModelKind,
        directory: &Path,
        threads: usize,
        variant: Variant,
    ) -> Result<Self> {
        match kind {
            ModelKind::Clip => Ok(Self::Clip(Box::new(clip::Clip::load(
                directory, threads, variant,
            )?))),
            ModelKind::Whisper => Ok(Self::Whisper(Box::new(whisper::Whisper::load(
                directory, threads, variant,
            )?))),
            ModelKind::Depth => Ok(Self::Depth(Box::new(depth::Depth::load(
                directory, threads, variant,
            )?))),
        }
    }

    pub fn clip_rank(
        &mut self,
        image: &DynamicImage,
        labels: &[String],
        top_k: usize,
        truncate_text: bool,
        cancel: &AtomicBool,
    ) -> Result<Vec<LabelScore>> {
        match self {
            Self::Clip(engine) => engine.rank(image, labels, top_k, truncate_text, cancel),
            _ => bail!("model does not support image/text ranking"),
        }
    }

    pub fn clip_image_embedding(&mut self, image: &DynamicImage) -> Result<Vec<f32>> {
        match self {
            Self::Clip(engine) => engine.image_embedding(image),
            _ => bail!("model does not support image embeddings"),
        }
    }

    pub fn clip_text_embedding(&mut self, text: &str, truncate_text: bool) -> Result<Vec<f32>> {
        match self {
            Self::Clip(engine) => engine.text_embedding(text, truncate_text),
            _ => bail!("model does not support text embeddings"),
        }
    }

    pub fn depth(
        &mut self,
        image: &DynamicImage,
        resolution: OutputResolution,
    ) -> Result<DepthMap> {
        match self {
            Self::Depth(engine) => engine.estimate(image, resolution),
            _ => bail!("model does not support depth estimation"),
        }
    }

    pub fn transcribe_wav(
        &mut self,
        wav: &[u8],
        language: &str,
        max_output_tokens: usize,
        cancel: &AtomicBool,
    ) -> Result<Transcription> {
        match self {
            Self::Whisper(engine) => {
                engine.transcribe_wav(wav, language, max_output_tokens, cancel)
            }
            _ => bail!("model does not support transcription"),
        }
    }
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    if !(1..=256).contains(&threads) {
        bail!("threads must be between 1 and 256");
    }
    if !path.is_file() {
        bail!("model asset missing: {}", path.display());
    }
    Session::builder()?
        .with_intra_threads(threads)
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .commit_from_file(path)
        .with_context(|| format!("load ONNX graph {}", path.display()))
}

fn graph(directory: &Path, stem: &str, variant: Variant) -> PathBuf {
    directory.join("onnx").join(variant.filename(stem))
}

/// The Depth Anything export does not follow the `_int8` file convention: its
/// 4-bit `MatMulNBits` weights are published as `model_quantized.onnx`. An ONNX
/// external-data graph names its weights file internally, so this pair cannot be
/// renamed to a precision-neutral name — the loader has to use the published
/// names for both the graph and its `_data` sibling.
fn depth_graph(directory: &Path, variant: Variant) -> PathBuf {
    directory.join("onnx").join(match variant {
        Variant::Fp32 => "model.onnx",
        Variant::Int8 => "model_quantized.onnx",
    })
}

fn read_json(directory: &Path, name: &str) -> Result<serde_json::Value> {
    let path = directory.join(name);
    serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))
}

fn vector_output(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>> {
    let output = outputs
        .get(name)
        .with_context(|| format!("model output {name} missing"))?;
    let (shape, data) = output.try_extract_tensor::<f32>()?;
    validate_vector(shape, data, name, expected)?;
    Ok(data.to_vec())
}

fn validate_vector(shape: &[i64], data: &[f32], name: &str, expected: usize) -> Result<()> {
    if !matches!(shape, [n] if *n == expected as i64)
        && !matches!(shape, [1, n] if *n == expected as i64)
    {
        bail!("invalid {name} embedding shape: {shape:?}");
    }
    if data.len() != expected || data.iter().any(|value| !value.is_finite()) {
        bail!("invalid {name} embedding: expected {expected} finite elements");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_output_requires_one_vector_not_just_matching_element_count() {
        let values = [0.0; 4];
        for shape in [&[4][..], &[1, 4][..]] {
            assert!(validate_vector(shape, &values, "embedding", 4).is_ok());
        }
        for shape in [&[2, 2][..], &[1, 1, 4][..], &[4, 1][..]] {
            assert!(validate_vector(shape, &values, "embedding", 4).is_err());
        }
        assert!(validate_vector(&[4], &[f32::NAN; 4], "embedding", 4).is_err());
        assert!(validate_vector(&[4], &[0.0; 3], "embedding", 4).is_err());
    }

    #[test]
    fn depth_graph_names_track_the_published_export() {
        let root = Path::new("/models/depth");
        assert_eq!(
            depth_graph(root, Variant::Fp32),
            root.join("onnx/model.onnx")
        );
        assert_eq!(
            depth_graph(root, Variant::Int8),
            root.join("onnx/model_quantized.onnx")
        );
        // The loader derives the external-data sibling from this name, so the
        // pair has to agree with what the graph references internally.
        let data = depth_graph(root, Variant::Int8).with_file_name("model_quantized.onnx_data");
        assert_eq!(data, root.join("onnx/model_quantized.onnx_data"));
        // Models that do follow the convention are unaffected.
        assert_eq!(
            graph(root, "vision_model", Variant::Int8),
            root.join("onnx/vision_model_int8.onnx")
        );
    }
}
