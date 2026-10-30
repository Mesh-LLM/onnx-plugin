//! Image-to-tensor preparation for vision adapters.

use anyhow::{Result, bail};
use image::{DynamicImage, RgbImage, imageops::FilterType};

pub(super) fn validate_image(image: &DynamicImage) -> Result<()> {
    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > 40_000_000 {
        bail!("image must contain between 1 and 40000000 pixels");
    }
    Ok(())
}

pub(super) fn center_crop(image: &DynamicImage, size: u32) -> Result<RgbImage> {
    validate_image(image)?;
    let rgb = image.to_rgb8();
    let scale = f64::from(size) / f64::from(rgb.width().min(rgb.height()));
    let width = (f64::from(rgb.width()) * scale)
        .floor()
        .max(f64::from(size)) as u32;
    let height = (f64::from(rgb.height()) * scale)
        .floor()
        .max(f64::from(size)) as u32;
    let resized = image::imageops::resize(&rgb, width, height, FilterType::CatmullRom);
    Ok(image::imageops::crop_imm(
        &resized,
        (width - size) / 2,
        (height - size) / 2,
        size,
        size,
    )
    .to_image())
}

pub(super) fn normalized_chw(image: &RgbImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let plane = image.width() as usize * image.height() as usize;
    let mut data = vec![0.0; plane * 3];
    for (index, pixel) in image.pixels().enumerate() {
        for channel in 0..3 {
            data[channel * plane + index] =
                (f32::from(pixel[channel]) / 255.0 - mean[channel]) / std[channel];
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn crop_preserves_center_and_chw_channel_order() {
        let mut rgb = RgbImage::from_pixel(4, 2, Rgb([255, 0, 0]));
        for x in 1..3 {
            for y in 0..2 {
                rgb.put_pixel(x, y, Rgb([0, 255, 0]));
            }
        }
        let cropped = center_crop(&DynamicImage::ImageRgb8(rgb), 2).unwrap();
        assert_eq!(cropped.get_pixel(0, 0), &Rgb([0, 255, 0]));
        let chw = normalized_chw(&cropped, [0.0; 3], [1.0; 3]);
        assert_eq!(&chw[..4], &[0.0; 4]);
        assert_eq!(&chw[4..8], &[1.0; 4]);
    }

    #[test]
    fn rejects_empty_images() {
        assert!(center_crop(&DynamicImage::new_rgb8(0, 0), 224).is_err());
    }
}
