//! Bounded document decoding, PDF text extraction, and raster OCR.

use std::{
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use image::GrayImage;
use pdfium_render::prelude::*;
use serde::{Deserialize, Serialize};

use crate::ocr::{Document, Engine, Line, Page, document::*};

const MAX_DIMENSION: i32 = 4096;
const MAX_PAGES: usize = 1000;
const MAX_IMAGE_REGIONS: usize = 256;
static PDFIUM_BIND_LOCK: Mutex<()> = Mutex::new(());

/// Errors at the document boundary, distinct from model inference failures.
#[derive(Debug)]
pub enum InputError {
    Invalid(String),
    Unavailable(String),
}

impl std::fmt::Display for InputError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) | Self::Unavailable(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for InputError {}

fn invalid(error: impl std::fmt::Display) -> anyhow::Error {
    InputError::Invalid(error.to_string()).into()
}

fn check_cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("document processing cancelled");
    }
    Ok(())
}

/// Request defaults and per-document overrides. Passwords are never Debug output.
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PdfOptions {
    pub dpi: u32,
    pub mode: OcrMode,
    pub image_ocr: bool,
    pub keep_furniture: bool,
    pub max_pages: usize,
    #[serde(skip_serializing)]
    pub password: Option<String>,
}

impl Default for PdfOptions {
    fn default() -> Self {
        Self {
            dpi: 200,
            mode: OcrMode::Auto,
            image_ocr: true,
            keep_furniture: false,
            max_pages: MAX_PAGES,
            password: None,
        }
    }
}

impl PdfOptions {
    /// Validate hard processing bounds before opening documents or model sessions.
    pub fn validate(&self) -> Result<()> {
        if !(36..=600).contains(&self.dpi) {
            bail!("dpi must be between 36 and 600");
        }
        if !(1..=MAX_PAGES).contains(&self.max_pages) {
            bail!("max_pages must be between 1 and 1000");
        }
        if self
            .password
            .as_ref()
            .is_some_and(|password| password.len() > 1024)
        {
            bail!("PDF password exceeds supported length");
        }
        Ok(())
    }
}

/// Decode a PDF or supported image. Text-only PDF processing needs no engine.
#[cfg(test)]
pub fn process(
    data: &[u8],
    library: Option<&Path>,
    options: &PdfOptions,
    engine: Option<&mut Engine>,
) -> Result<Document> {
    process_cancellable(data, library, options, engine, &AtomicBool::new(false))
}

/// Validate cheap document properties before acquiring a model lease.
/// PDF parsing and password checks remain Pdfium's responsibility during processing.
pub fn validate_input(data: &[u8], options: &PdfOptions) -> Result<()> {
    options.validate().map_err(invalid)?;
    if data.starts_with(b"%PDF-") {
        if data.len() < 20 || !data.windows(5).any(|window| window == b"%%EOF") {
            return Err(invalid("invalid or truncated PDF upload"));
        }
    } else {
        if options.mode == OcrMode::Text {
            return Err(invalid("text mode requires a PDF text layer"));
        }
        decode_image(data)?;
    }
    Ok(())
}

/// Process a document with cancellation checkpoints between pages and image regions.
pub fn process_cancellable(
    data: &[u8],
    library: Option<&Path>,
    options: &PdfOptions,
    mut engine: Option<&mut Engine>,
    cancel: &AtomicBool,
) -> Result<Document> {
    options.validate().map_err(invalid)?;
    check_cancelled(cancel)?;
    let pages = if data.starts_with(b"%PDF-") {
        process_pdf(data, library, options, &mut engine, cancel)?
    } else {
        if options.mode == OcrMode::Text {
            return Err(invalid("text mode requires a PDF text layer"));
        }
        let image = decode_image(data)?;
        vec![required_engine(&mut engine)?.recognize_image_with_options(
            &image,
            0,
            options.keep_furniture,
        )?]
    };
    Ok(Document {
        model: "datalab-to/indra-ocr@v1.1.0",
        pages,
    })
}

fn decode_image(data: &[u8]) -> Result<GrayImage> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(invalid)?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DIMENSION as u32);
    limits.max_image_height = Some(MAX_DIMENSION as u32);
    limits.max_alloc = Some(64 * 1024 * 1024);
    reader.limits(limits);
    Ok(reader.decode().map_err(invalid)?.into_luma8())
}

fn required_engine<'a>(engine: &'a mut Option<&mut Engine>) -> Result<&'a mut Engine> {
    engine
        .as_deref_mut()
        .context("OCR model is required for this document")
}

fn process_pdf(
    data: &[u8],
    library: Option<&Path>,
    options: &PdfOptions,
    engine: &mut Option<&mut Engine>,
    cancel: &AtomicBool,
) -> Result<Vec<Page>> {
    let lock = PDFIUM_BIND_LOCK
        .lock()
        .map_err(|_| anyhow::anyhow!("Pdfium binding lock poisoned"))?;
    let bindings = match library {
        Some(path) => Pdfium::bind_to_library(path),
        None => Pdfium::bind_to_system_library(),
    };
    let pdfium = match bindings {
        Ok(bindings) => Pdfium::new(bindings),
        Err(PdfiumError::PdfiumLibraryBindingsAlreadyInitialized) => Pdfium::default(),
        Err(error) => {
            return Err(InputError::Unavailable(format!("load Pdfium library: {error}")).into());
        }
    };
    drop(lock);
    // Never include the password or uploaded bytes in error context.
    let document = pdfium
        .load_pdf_from_byte_vec(data.to_vec(), options.password.as_deref())
        .map_err(|error| invalid(format!("open uploaded PDF: {error}")))?;
    let config = PdfRenderConfig::new()
        .scale_page_by_factor(options.dpi as f32 / 72.0)
        .set_maximum_width(MAX_DIMENSION)
        .set_maximum_height(MAX_DIMENSION);
    document
        .pages()
        .iter()
        .take(options.max_pages)
        .enumerate()
        .map(|(index, page)| {
            check_cancelled(cancel)?;
            process_page(&page, index, &config, options, engine, cancel)
        })
        .collect()
}

fn process_page(
    page: &PdfPage<'_>,
    index: usize,
    config: &PdfRenderConfig,
    options: &PdfOptions,
    engine: &mut Option<&mut Engine>,
    cancel: &AtomicBool,
) -> Result<Page> {
    let image = page
        .render_with_config(config)
        .context("render PDF page")?
        .as_image()
        .context("decode PDF page bitmap")?
        .into_luma8();
    if options.mode == OcrMode::Ocr {
        return required_engine(engine)?.recognize_image_with_options(
            &image,
            index,
            options.keep_furniture,
        );
    }
    let mut lines = text_lines(page, config, image.width(), image.height())?;
    if options.mode == OcrMode::Auto && !text_layer_usable(&lines, image.width(), image.height()) {
        return required_engine(engine)?.recognize_image_with_options(
            &image,
            index,
            options.keep_furniture,
        );
    }
    let mut hybrid = false;
    if options.image_ocr {
        let regions = image_regions(page, config, image.width(), image.height())?;
        if !regions.is_empty() {
            let engine = required_engine(engine)?;
            for region in regions {
                check_cancelled(cancel)?;
                append_image_ocr(&image, region, &mut lines, engine, index, options)?;
                hybrid = true;
            }
        }
    }
    if hybrid {
        lines.sort_by(|a, b| {
            a.bbox[1]
                .total_cmp(&b.bbox[1])
                .then(a.bbox[0].total_cmp(&b.bbox[0]))
        });
        page_from_hybrid_lines(
            lines,
            index,
            image.width(),
            image.height(),
            options.keep_furniture,
        )
    } else {
        page_from_text_lines(
            lines,
            index,
            image.width(),
            image.height(),
            options.keep_furniture,
        )
    }
}

fn text_lines(
    page: &PdfPage<'_>,
    config: &PdfRenderConfig,
    width: u32,
    height: u32,
) -> Result<Vec<Line>> {
    let text = page.text().context("read PDF text layer")?;
    text.segments()
        .iter()
        .filter_map(|segment| {
            let content = segment.text();
            (!content.trim().is_empty()).then_some((content, segment.bounds()))
        })
        .map(|(text, rect)| {
            Ok(
                rect_pixels(page, rect, config, width, height)?.map(|bbox| Line {
                    text,
                    bbox,
                    confidence: 1.0,
                }),
            )
        })
        .collect::<Result<Vec<_>>>()
        .map(|lines| lines.into_iter().flatten().collect())
}

fn rect_pixels(
    page: &PdfPage<'_>,
    rect: PdfRect,
    config: &PdfRenderConfig,
    width: u32,
    height: u32,
) -> Result<Option<[f32; 4]>> {
    let corners = [
        (rect.left(), rect.top()),
        (rect.right(), rect.top()),
        (rect.left(), rect.bottom()),
        (rect.right(), rect.bottom()),
    ]
    .map(|(x, y)| page.points_to_pixels(x, y, config));
    let mut bbox = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for corner in corners {
        let (x, y) = corner?;
        bbox[0] = bbox[0].min(x as f32);
        bbox[1] = bbox[1].min(y as f32);
        bbox[2] = bbox[2].max(x as f32);
        bbox[3] = bbox[3].max(y as f32);
    }
    bbox[0] = bbox[0].clamp(0.0, width as f32);
    bbox[1] = bbox[1].clamp(0.0, height as f32);
    bbox[2] = bbox[2].clamp(0.0, width as f32);
    bbox[3] = bbox[3].clamp(0.0, height as f32);
    Ok((bbox[2] > bbox[0] && bbox[3] > bbox[1]).then_some(bbox))
}

fn contains_image(object: &PdfPageObject<'_>, depth: usize) -> Result<bool> {
    if object.as_image_object().is_some() {
        return Ok(true);
    }
    if let Some(form) = object.as_x_object_form_object() {
        if depth >= 16 {
            bail!("PDF image nesting exceeds supported depth");
        }
        for index in form.as_range() {
            if contains_image(&form.get(index)?, depth + 1)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn image_regions(
    page: &PdfPage<'_>,
    config: &PdfRenderConfig,
    width: u32,
    height: u32,
) -> Result<Vec<[f32; 4]>> {
    let mut regions = Vec::new();
    for object in page.objects().iter() {
        if contains_image(&object, 0)? {
            if regions.len() >= MAX_IMAGE_REGIONS {
                bail!("PDF page exceeds 256 image regions");
            }
            if let Some(rect) =
                rect_pixels(page, object.bounds()?.to_rect(), config, width, height)?
            {
                regions.push(rect);
            }
        }
    }
    Ok(regions)
}

fn append_image_ocr(
    page: &GrayImage,
    region: [f32; 4],
    lines: &mut Vec<Line>,
    engine: &mut Engine,
    index: usize,
    options: &PdfOptions,
) -> Result<()> {
    let [x0, y0, x1, y1] = region.map(|value| value as u32);
    if x1 <= x0 || y1 <= y0 {
        return Ok(());
    }
    let crop = image::imageops::crop_imm(page, x0, y0, x1 - x0, y1 - y0).to_image();
    let result = engine.recognize_image_with_options(&crop, index, options.keep_furniture)?;
    for mut line in result.lines {
        line.bbox = restore_crop_box(
            line.bbox,
            crop.width(),
            crop.height(),
            result.rotation,
            result.vertical,
        );
        line.bbox[0] += x0 as f32;
        line.bbox[2] += x0 as f32;
        line.bbox[1] += y0 as f32;
        line.bbox[3] += y0 as f32;
        if !lines
            .iter()
            .any(|prior| prior.text.trim() == line.text.trim() && overlaps(prior.bbox, line.bbox))
        {
            lines.push(line);
        }
    }
    Ok(())
}

fn overlaps(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

fn restore_crop_box(
    bbox: [f32; 4],
    width: u32,
    height: u32,
    rotation: u16,
    vertical: bool,
) -> [f32; 4] {
    let rotation = if vertical { 270 } else { rotation };
    let inverse = |x: f32, y: f32| match rotation {
        90 => (y, height as f32 - x),
        180 => (width as f32 - x, height as f32 - y),
        270 => (width as f32 - y, x),
        _ => (x, y),
    };
    let (x0, y0) = inverse(bbox[0], bbox[1]);
    let (x1, y1) = inverse(bbox[2], bbox[3]);
    [
        x0.min(x1).clamp(0.0, width as f32),
        y0.min(y1).clamp(0.0, height as f32),
        x0.max(x1).clamp(0.0, width as f32),
        y0.max(y1).clamp(0.0, height as f32),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processing_bounds_and_secret_serialization() {
        let mut options = PdfOptions {
            password: Some("do-not-persist".into()),
            ..PdfOptions::default()
        };
        assert!(
            !serde_json::to_string(&options)
                .unwrap()
                .contains("do-not-persist")
        );
        options.dpi = 0;
        assert!(options.validate().is_err());
        options.dpi = 200;
        options.max_pages = 0;
        assert!(options.validate().is_err());
        assert!(serde_json::from_str::<PdfOptions>(r#"{"unexpected":true}"#).is_err());
    }

    #[test]
    fn malformed_input_is_typed_and_cancellation_precedes_parsing() {
        let options = PdfOptions::default();
        for bytes in [&b"corrupt image"[..], &b"%PDF-1.7 truncated"[..]] {
            let error = validate_input(bytes, &options).expect_err("invalid input");
            assert!(matches!(
                error.downcast_ref::<InputError>(),
                Some(InputError::Invalid(_))
            ));
        }
        let error = process_cancellable(b"invalid", None, &options, None, &AtomicBool::new(true))
            .expect_err("cancelled");
        assert!(error.to_string().contains("cancelled"));
    }

    #[test]
    fn rotated_region_boxes_map_back_to_page_pixels() {
        assert_eq!(
            restore_crop_box([10.0, 20.0, 30.0, 40.0], 100, 200, 90, false),
            [20.0, 170.0, 40.0, 190.0]
        );
        assert_eq!(
            restore_crop_box([10.0, 20.0, 30.0, 40.0], 100, 200, 0, true),
            [60.0, 10.0, 80.0, 30.0]
        );
    }

    #[test]
    fn text_mode_rejects_raster_inputs_before_loading_an_engine() {
        let options = PdfOptions {
            mode: OcrMode::Text,
            ..PdfOptions::default()
        };
        assert!(
            process(b"not-a-pdf", None, &options, None)
                .unwrap_err()
                .to_string()
                .contains("requires a PDF")
        );
    }

    fn text_pdf() -> Vec<u8> {
        use std::fmt::Write;
        let content = "BT /F1 12 Tf 72 720 Td (A real PDF text layer without model weights.) Tj ET";
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_string(),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
            format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len()),
        ];
        let mut output = "%PDF-1.7\n".to_string();
        let mut offsets = vec![0];
        for (index, object) in objects.iter().enumerate() {
            offsets.push(output.len());
            writeln!(output, "{} 0 obj\n{object}\nendobj", index + 1).unwrap();
        }
        let xref = output.len();
        writeln!(output, "xref\n0 {}\n0000000000 65535 f ", offsets.len()).unwrap();
        for offset in &offsets[1..] {
            writeln!(output, "{offset:010} 00000 n ").unwrap();
        }
        writeln!(
            output,
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF",
            offsets.len()
        )
        .unwrap();
        output.into_bytes()
    }

    #[test]
    #[ignore = "requires ONNX_OCR_MODEL_DIR and PDFIUM_LIBRARY; renders and recognizes a real scan"]
    fn real_pdf_ocrs_a_scanned_document_end_to_end() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let models = std::env::var("ONNX_OCR_MODEL_DIR").expect("ONNX_OCR_MODEL_DIR");
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/scan.pdf"
        ))
        .unwrap();
        let options = PdfOptions {
            mode: OcrMode::Auto,
            ..PdfOptions::default()
        };
        let mut engine = crate::ocr::Engine::load(Path::new(&models)).expect("load ONNX sessions");
        let document = process(
            &data,
            Some(Path::new(&library)),
            &options,
            Some(&mut engine),
        )
        .unwrap();
        assert_eq!(document.pages.len(), 1);
        let page = &document.pages[0];
        // The fixture is a photographed letter with no text layer, so `auto`
        // must choose recognition rather than trusting an empty layer.
        assert_eq!(page.source, "ocr");
        assert!(!page.lines.is_empty());
        assert!(page.text.contains("SEOUL"), "{}", page.text);
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY pointing to a native Pdfium library; no model weights needed"]
    fn real_pdf_text_mode_needs_no_model_and_dpi_scales_coordinates() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let mut options = PdfOptions {
            dpi: 72,
            mode: OcrMode::Text,
            image_ocr: false,
            ..PdfOptions::default()
        };
        let first = process(&text_pdf(), Some(Path::new(&library)), &options, None).unwrap();
        assert_eq!(first.pages.len(), 1);
        assert_eq!(first.pages[0].source, "text");
        assert!(first.pages[0].text.contains("without model weights"));
        options.dpi = 144;
        let second = process(&text_pdf(), Some(Path::new(&library)), &options, None).unwrap();
        assert!(
            (second.pages[0].lines[0].bbox[0] - 2.0 * first.pages[0].lines[0].bbox[0]).abs() <= 2.0
        );
    }
}
