//! Indra page orientation classifier and quarter-turn correction.

use anyhow::{Context, Result};
use image::{GrayImage, imageops::FilterType};
use ort::{inputs, session::Session, value::Tensor};

const SIDE: usize = 768;
const TURN_THRESHOLD: f32 = 0.95;

pub struct UprightPage {
    pub image: GrayImage,
    pub rotation: u16,
    pub vertical: bool,
}

pub fn orient(session: Option<&mut Session>, image: &GrayImage) -> Result<UprightPage> {
    let Some(session) = session else {
        return Ok(UprightPage {
            image: image.clone(),
            rotation: 0,
            vertical: false,
        });
    };
    let scale = SIDE as f32 / image.width().max(image.height()) as f32;
    let width = ((image.width() as f32 * scale).round() as u32).max(1);
    let height = ((image.height() as f32 * scale).round() as u32).max(1);
    let resized = image::imageops::resize(image, width, height, FilterType::Triangle);
    let mut data = vec![0.0_f32; SIDE * SIDE];
    for y in 0..height as usize {
        for x in 0..width as usize {
            data[y * SIDE + x] = 1.0 - resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
    let input = Tensor::from_array(([1, 1, SIDE, SIDE], data))?;
    let outputs = session.run(inputs!["x" => input])?;
    let (_, probabilities) = outputs["p"].try_extract_tensor::<f32>()?;
    let (class, probability) = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .context("orientation model returned no classes")?;
    let class = if *probability >= TURN_THRESHOLD {
        class
    } else {
        0
    };
    let (image, rotation, vertical) = match class {
        1 => (image::imageops::rotate90(image), 90, false),
        2 => (image::imageops::rotate180(image), 180, false),
        3 => (image::imageops::rotate270(image), 270, false),
        4 => (image::imageops::rotate270(image), 0, true),
        _ => (image.clone(), 0, false),
    };
    Ok(UprightPage {
        image,
        rotation,
        vertical,
    })
}
