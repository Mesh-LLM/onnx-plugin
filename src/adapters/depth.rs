//! Depth Anything V2 Small dense relative-depth inference.

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::{DynamicImage, GrayImage, imageops::FilterType};
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};

use super::{Variant, depth_graph, pixels, read_json, session};

const MAX_OUTPUT_PIXELS: usize = 4_000_000;

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputResolution {
    #[default]
    Input,
    Native,
}

/// Finite row-major relative depth. Values are not calibrated metric distance.
#[derive(Debug, Clone, Serialize)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub units: &'static str,
    pub values: Vec<f32>,
}

impl DepthMap {
    /// Produce a display-only grayscale preview; numerical values remain unchanged.
    pub fn normalized_preview(&self) -> Result<GrayImage> {
        validate_map(self.width, self.height, &self.values)?;
        let min = self.values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let pixels = self
            .values
            .iter()
            .map(|value| {
                if range <= f32::EPSILON {
                    0
                } else {
                    (((value - min) / range) * 255.0).round() as u8
                }
            })
            .collect();
        GrayImage::from_raw(self.width, self.height, pixels).context("invalid depth preview size")
    }
}

pub struct Depth {
    session: Session,
    processor: Processor,
}

struct Processor {
    width: u32,
    height: u32,
    multiple: u32,
    keep_aspect_ratio: bool,
    mean: [f32; 3],
    std: [f32; 3],
}

impl Depth {
    pub fn load(directory: &Path, threads: usize, variant: Variant) -> Result<Self> {
        let path = depth_graph(directory, variant);
        let data_path = path.with_file_name(format!(
            "{}_data",
            path.file_name()
                .context("invalid depth graph path")?
                .to_string_lossy()
        ));
        if !data_path.is_file() {
            bail!(
                "Depth external tensor data missing: {}",
                data_path.display()
            );
        }
        Ok(Self {
            session: session(&path, threads)?,
            processor: Processor::load(directory)?,
        })
    }

    pub fn estimate(
        &mut self,
        image: &DynamicImage,
        resolution: OutputResolution,
    ) -> Result<DepthMap> {
        pixels::validate_image(image)?;
        if resolution == OutputResolution::Input {
            checked_pixels(image.width(), image.height())?;
        }
        let (width, height) = self.processor.network_size(image.width(), image.height())?;
        let resized =
            image::imageops::resize(&image.to_rgb8(), width, height, FilterType::CatmullRom);
        let pixels = pixels::normalized_chw(&resized, self.processor.mean, self.processor.std);
        let inputs = ort::inputs!["pixel_values" => Tensor::from_array(([1, 3, height as usize, width as usize], pixels))?];
        let outputs = self.session.run(inputs)?;
        let value = outputs
            .get("predicted_depth")
            .context("Depth output predicted_depth missing")?;
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 {
            bail!("expected Depth output shape [1, height, width]");
        }
        let output_height = u32::try_from(shape[1]).context("invalid Depth output height")?;
        let output_width = u32::try_from(shape[2]).context("invalid Depth output width")?;
        validate_map(output_width, output_height, data)?;
        let (target_width, target_height) = match resolution {
            OutputResolution::Input => (image.width(), image.height()),
            OutputResolution::Native => (output_width, output_height),
        };
        let values = resize_bilinear(
            data,
            output_width,
            output_height,
            target_width,
            target_height,
        )?;
        Ok(DepthMap {
            width: target_width,
            height: target_height,
            units: "relative",
            values,
        })
    }
}

fn checked_pixels(width: u32, height: u32) -> Result<usize> {
    let count = u64::from(width) * u64::from(height);
    if count == 0 || count > MAX_OUTPUT_PIXELS as u64 {
        bail!("depth map must contain between 1 and {MAX_OUTPUT_PIXELS} pixels");
    }
    Ok(count as usize)
}

fn validate_map(width: u32, height: u32, values: &[f32]) -> Result<()> {
    if checked_pixels(width, height)? != values.len()
        || values.iter().any(|value| !value.is_finite())
    {
        bail!("depth map has invalid shape or nonfinite values");
    }
    Ok(())
}

impl Processor {
    fn load(directory: &Path) -> Result<Self> {
        let config = read_json(directory, "preprocessor_config.json")?;
        let width = u32::try_from(
            config["size"]["width"]
                .as_u64()
                .context("Depth processor width missing")?,
        )?;
        let height = u32::try_from(
            config["size"]["height"]
                .as_u64()
                .context("Depth processor height missing")?,
        )?;
        let multiple = u32::try_from(config["ensure_multiple_of"].as_u64().unwrap_or(1))?;
        if width == 0
            || height == 0
            || width > 2048
            || height > 2048
            || multiple == 0
            || multiple > width.min(height)
        {
            bail!("invalid Depth processor dimensions");
        }
        let channels = |name: &str| -> Result<[f32; 3]> {
            let array = config[name]
                .as_array()
                .with_context(|| format!("Depth processor {name} missing"))?;
            if array.len() != 3 {
                bail!("Depth processor {name} must have three channels");
            }
            let values = array
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .map(|value| value as f32)
                        .filter(|value| value.is_finite())
                        .context("invalid Depth normalization value")
                })
                .collect::<Result<Vec<_>>>()?;
            values
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid Depth normalization channels"))
        };
        let std = channels("image_std")?;
        if std.iter().any(|value| *value <= 0.0) {
            bail!("Depth processor standard deviation must be positive");
        }
        Ok(Self {
            width,
            height,
            multiple,
            keep_aspect_ratio: config["keep_aspect_ratio"].as_bool().unwrap_or(false),
            mean: channels("image_mean")?,
            std,
        })
    }

    fn network_size(&self, width: u32, height: u32) -> Result<(u32, u32)> {
        let mut scale_width = f64::from(self.width) / f64::from(width);
        let mut scale_height = f64::from(self.height) / f64::from(height);
        // Hugging Face DPT preserves aspect using the resize factor closest to one.
        if self.keep_aspect_ratio {
            if (1.0 - scale_width).abs() < (1.0 - scale_height).abs() {
                scale_height = scale_width;
            } else {
                scale_width = scale_height;
            }
        }
        let constrain = |dimension: f64| -> Result<u32> {
            let rounded =
                (dimension / f64::from(self.multiple)).round().max(1.0) * f64::from(self.multiple);
            if !rounded.is_finite() || rounded > MAX_OUTPUT_PIXELS as f64 {
                bail!("Depth processor dimensions exceed output bounds");
            }
            Ok(rounded as u32)
        };
        let size = (
            constrain(f64::from(width) * scale_width)?,
            constrain(f64::from(height) * scale_height)?,
        );
        checked_pixels(size.0, size.1)?;
        Ok(size)
    }
}

fn resize_bilinear(
    values: &[f32],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<f32>> {
    validate_map(width, height, values)?;
    let count = checked_pixels(target_width, target_height)?;
    if (width, height) == (target_width, target_height) {
        return Ok(values.to_vec());
    }
    let mut output = Vec::with_capacity(count);
    for y in 0..target_height {
        let source_y = ((y as f64 + 0.5) * f64::from(height) / f64::from(target_height) - 0.5)
            .clamp(0.0, f64::from(height - 1));
        let y0 = source_y.floor() as u32;
        let y1 = (y0 + 1).min(height - 1);
        let fy = (source_y - f64::from(y0)) as f32;
        for x in 0..target_width {
            let source_x = ((x as f64 + 0.5) * f64::from(width) / f64::from(target_width) - 0.5)
                .clamp(0.0, f64::from(width - 1));
            let x0 = source_x.floor() as u32;
            let x1 = (x0 + 1).min(width - 1);
            let fx = (source_x - f64::from(x0)) as f32;
            let at = |x, y| values[(y * width + x) as usize];
            let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
            let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
            output.push(top * (1.0 - fy) + bottom * fy);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_preserves_grid_alignment_and_center_value() {
        let resized = resize_bilinear(&[0.0, 2.0, 2.0, 4.0], 2, 2, 3, 3).unwrap();
        assert_eq!(resized[0], 0.0);
        assert_eq!(resized[4], 2.0);
        assert_eq!(resized[8], 4.0);
    }

    #[test]
    fn preview_is_display_only_and_constant_maps_are_valid() {
        let map = DepthMap {
            width: 2,
            height: 1,
            units: "relative",
            values: vec![10.0, 20.0],
        };
        assert_eq!(map.normalized_preview().unwrap().as_raw(), &[0, 255]);
        assert_eq!(map.values, [10.0, 20.0]);
        let constant = DepthMap {
            values: vec![3.0, 3.0],
            ..map
        };
        assert_eq!(constant.normalized_preview().unwrap().as_raw(), &[0, 0]);
    }

    #[test]
    fn rejects_invalid_shapes_and_unbounded_output() {
        assert!(resize_bilinear(&[f32::NAN], 1, 1, 1, 1).is_err());
        assert!(resize_bilinear(&[1.0], 1, 1, 10_000, 10_000).is_err());
        let processor = Processor {
            width: 518,
            height: 518,
            multiple: 14,
            keep_aspect_ratio: true,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        assert_eq!(processor.network_size(1024, 512).unwrap(), (1036, 518));
    }

    #[test]
    #[ignore = "requires a pinned Depth bundle; set ONNX_DEPTH_MODEL_DIR and/or ONNX_DEPTH_INT8_MODEL_DIR"]
    fn depth_runs_native_graph_and_external_data() {
        let image = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/scan.png"
        ))
        .unwrap();
        let mut maps = Vec::new();
        for (variable, variant) in [
            ("ONNX_DEPTH_MODEL_DIR", Variant::Fp32),
            ("ONNX_DEPTH_INT8_MODEL_DIR", Variant::Int8),
        ] {
            let Ok(dir) = std::env::var(variable) else {
                continue;
            };
            let mut engine = Depth::load(Path::new(&dir), 1, variant).unwrap();
            let depth = engine.estimate(&image, OutputResolution::Native).unwrap();
            assert_eq!(depth.units, "relative");
            assert!(!depth.values.is_empty());
            assert!(depth.values.iter().all(|value| value.is_finite()));
            // The preview must be renderable from whatever the graph produced,
            // including the 4-bit `MatMulNBits` export.
            assert_eq!(
                depth.normalized_preview().unwrap().dimensions(),
                (depth.width, depth.height)
            );
            maps.push((variable, depth));
        }
        assert!(
            !maps.is_empty(),
            "set ONNX_DEPTH_MODEL_DIR and/or ONNX_DEPTH_INT8_MODEL_DIR"
        );
        // The quantized graph must describe the same scene: identical geometry
        // and a strongly correlated depth ordering.
        if let [(_, fp32), (_, int8)] = maps.as_slice() {
            assert_eq!((fp32.width, fp32.height), (int8.width, int8.height));
            assert!(
                correlation(&fp32.values, &int8.values) > 0.9,
                "fp32 and int8 depth maps disagree"
            );
        }
    }

    /// Pearson correlation over two equally sized maps.
    fn correlation(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        let count = left.len() as f32;
        let mean = |values: &[f32]| values.iter().sum::<f32>() / count;
        let (left_mean, right_mean) = (mean(left), mean(right));
        let covariance: f32 = left
            .iter()
            .zip(right)
            .map(|(left, right)| (left - left_mean) * (right - right_mean))
            .sum();
        let left_spread: f32 = left.iter().map(|value| (value - left_mean).powi(2)).sum();
        let right_spread: f32 = right.iter().map(|value| (value - right_mean).powi(2)).sum();
        covariance / (left_spread * right_spread).sqrt()
    }
}
