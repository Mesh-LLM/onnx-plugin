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
    let side = image.width().min(image.height());
    let crop = image
        .crop_imm(
            (image.width() - side) / 2,
            (image.height() - side) / 2,
            side,
            side,
        )
        .to_rgb8();
    Ok(image::imageops::resize(
        &crop,
        size,
        size,
        FilterType::CatmullRom,
    ))
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
    fn thin_images_do_not_expand_the_long_axis_before_cropping() {
        // Resizing the short edge first would allocate a 224 x 224,000 strip.
        // Only the square central pixel contributes to this input.
        let mut image = RgbImage::from_pixel(1000, 1, Rgb([255, 0, 0]));
        image.put_pixel(499, 0, Rgb([0, 255, 0]));
        let cropped = center_crop(&DynamicImage::ImageRgb8(image), 224).unwrap();
        assert_eq!(cropped.dimensions(), (224, 224));
        assert!(cropped.pixels().all(|pixel| *pixel == Rgb([0, 255, 0])));
    }

    #[test]
    fn rejects_empty_images() {
        assert!(center_crop(&DynamicImage::new_rgb8(0, 0), 224).is_err());
    }
}
