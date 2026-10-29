//! Indra LineNet input and four-connected component decoding.

use anyhow::{Context, Result};
use image::{GrayImage, imageops::FilterType};
use ort::{inputs, session::Session, value::Tensor};

const STRIDE: f32 = 2.0;

pub fn detect_lines(
    session: &mut Session,
    page: &GrayImage,
    long_side: usize,
) -> Result<Vec<[f32; 4]>> {
    let height = page.height() as usize;
    let width = page.width() as usize;
    let scale = long_side as f32 / height.max(width) as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(1);
    let resized_height = ((height as f32 * scale).round() as u32).max(1);
    let resized =
        image::imageops::resize(page, resized_width, resized_height, FilterType::Triangle);
    let padded_width = (resized_width as usize).div_ceil(32) * 32;
    let padded_height = (resized_height as usize).div_ceil(32) * 32;
    let mut data = vec![0.0_f32; padded_width * padded_height];
    for y in 0..resized_height as usize {
        for x in 0..resized_width as usize {
            data[y * padded_width + x] =
                1.0 - resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
    let input = Tensor::from_array(([1, 1, padded_height, padded_width], data))?;
    let outputs = session.run(inputs!["x" => input])?;
    let (shape, logits) = outputs["out"].try_extract_tensor::<f32>()?;
    let out_height = usize::try_from(shape[2]).context("invalid detector height")?;
    let out_width = usize::try_from(shape[3]).context("invalid detector width")?;
    let channel_size = out_height * out_width;
    let mut visited = vec![false; channel_size];
    let mut boxes = Vec::new();
    for start in 0..channel_size {
        if visited[start] || logits[start] <= 0.0 {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut count = 0usize;
        let mut left = out_width;
        let mut right = 0usize;
        let mut top_sum = 0.0;
        let mut bottom_sum = 0.0;
        while let Some(index) = stack.pop() {
            count += 1;
            let y = index / out_width;
            let x = index % out_width;
            left = left.min(x);
            right = right.max(x);
            let center_y = (y as f32 + 0.5) * STRIDE;
            top_sum += center_y - logits[channel_size + index];
            bottom_sum += center_y + logits[2 * channel_size + index];
            for neighbor in neighbors(index, x, y, out_width, out_height) {
                if !visited[neighbor] && logits[neighbor] > 0.0 {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        if count < 2 {
            continue;
        }
        let y0 = top_sum / count as f32;
        let y1 = bottom_sum / count as f32;
        let line_height = (y1 - y0).max(1.0);
        let extension = 0.15 * line_height;
        boxes.push([
            ((left as f32 * STRIDE - extension) / scale).clamp(0.0, width as f32),
            (y0 / scale).clamp(0.0, height as f32),
            (((right + 1) as f32 * STRIDE + extension) / scale).clamp(0.0, width as f32),
            (y1 / scale).clamp(0.0, height as f32),
        ]);
    }
    Ok(boxes)
}

fn neighbors(
    index: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = usize> {
    [
        (x > 0).then_some(index.wrapping_sub(1)),
        (x + 1 < width).then_some(index + 1),
        (y > 0).then_some(index.wrapping_sub(width)),
        (y + 1 < height).then_some(index + width),
    ]
    .into_iter()
    .flatten()
}
