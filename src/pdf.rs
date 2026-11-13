//! Bounded document decoding, PDF text extraction, and raster OCR.

use std::{
    io::{Read, Seek},
    path::{Path, PathBuf},
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
const MAX_PAGE_OBJECTS: usize = 16_384;

#[cfg(test)]
thread_local! { static RASTERIZATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
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
    /// 1-based inclusive selection (`3`, `1-10`, `1-`); mutually exclusive
    /// with `max_pages`, which caps instead of selecting.
    pub pages: Option<PageRange>,
    /// Decoding-only metadata response: page count and rendered page sizes.
    pub pages_only: bool,
    /// Select NDJSON streaming on the canonical OCR route.
    pub stream: bool,
    /// Per-request acquisition + inference budget in seconds.
    pub deadline_secs: Option<u64>,
    #[serde(skip_serializing)]
    pub password: Option<String>,
}

/// 1-based inclusive page selection: `3` is one page, `1-10` a closed range,
/// `1-` runs to the end of the document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageRange {
    pub from: usize,
    pub to: Option<usize>,
}

impl PageRange {
    pub fn parse(input: &str) -> Result<Self> {
        let value = input.trim();
        let syntax =
            "expected a page number ('3'), an inclusive range ('1-10'), or an open range ('1-')";
        let invalid = || anyhow::anyhow!("invalid `pages` value '{input}': {syntax}");
        if value.is_empty() {
            return Err(invalid());
        }
        let parse_page = |text: &str| -> Result<usize> {
            let page: usize = text.trim().parse().map_err(|_| invalid())?;
            if page == 0 {
                bail!("invalid `pages` value '{input}': page numbers start at 1");
            }
            Ok(page)
        };
        match value.split_once('-') {
            None => {
                let from = parse_page(value)?;
                Ok(Self {
                    from,
                    to: Some(from),
                })
            }
            Some((start, "")) => Ok(Self {
                from: parse_page(start)?,
                to: None,
            }),
            Some((start, end)) => {
                let from = parse_page(start)?;
                let to = parse_page(end)?;
                if to < from {
                    bail!("invalid `pages` value '{input}': range end precedes its start");
                }
                Ok(Self { from, to: Some(to) })
            }
        }
    }

    /// Pages this selection covers in a document with `page_count` pages.
    /// Call [`Self::validate_against`] first.
    pub fn selected_len(&self, page_count: usize) -> usize {
        self.to.unwrap_or(page_count) - self.from + 1
    }

    pub fn validate_against(&self, page_count: usize) -> Result<()> {
        let to = self.to.unwrap_or(page_count);
        if self.from > page_count || to > page_count {
            bail!("document has {page_count} pages; pages {self} is out of range");
        }
        Ok(())
    }
}

impl std::fmt::Display for PageRange {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.to {
            Some(to) if to == self.from => write!(formatter, "{}", self.from),
            Some(to) => write!(formatter, "{}-{}", self.from, to),
            None => write!(formatter, "{}-", self.from),
        }
    }
}

impl<'de> Deserialize<'de> for PageRange {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(serde::de::Error::custom)
    }
}

impl Serialize for PageRange {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl Default for PdfOptions {
    fn default() -> Self {
        Self {
            dpi: 200,
            mode: OcrMode::Auto,
            image_ocr: true,
            keep_furniture: false,
            max_pages: MAX_PAGES,
            pages: None,
            pages_only: false,
            stream: false,
            deadline_secs: None,
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
        if let Some(deadline) = self.deadline_secs
            && !(30..=900).contains(&deadline)
        {
            bail!("deadline_secs must be between 30 and 900; {deadline} was requested");
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

/// Where document bytes come from. Spooled uploads stay on disk: Pdfium reads
/// PDFs from a path and image decoders take any reader, so the plugin only
/// buffers a document when a caller hands it bytes directly.
pub enum DocumentSource {
    /// Byte-vector entry point used by tests; the server always spools to a path.
    #[cfg(test)]
    Memory(Vec<u8>),
    Path(PathBuf),
}

impl DocumentSource {
    fn is_pdf(&self) -> Result<bool> {
        const MAGIC: &[u8; 5] = b"%PDF-";
        match self {
            #[cfg(test)]
            Self::Memory(data) => Ok(data.starts_with(MAGIC)),
            Self::Path(path) => {
                let mut file = std::fs::File::open(path)
                    .with_context(|| format!("open {}", path.display()))?;
                let mut head = [0u8; 5];
                let read = file
                    .read(&mut head)
                    .with_context(|| format!("read {}", path.display()))?;
                Ok(head[..read] == *MAGIC)
            }
        }
    }

    /// Cheap structural check so corrupt uploads fail before Pdfium is asked
    /// to interpret them (a valid envelope that still cannot be opened is
    /// treated as encrypted in `open_document`).
    fn validate_pdf_envelope(&self) -> Result<()> {
        let tail = match self {
            #[cfg(test)]
            Self::Memory(data) => data.clone(),
            Self::Path(path) => {
                let mut file = std::fs::File::open(path)
                    .with_context(|| format!("open {}", path.display()))?;
                let length = file
                    .metadata()
                    .with_context(|| format!("stat {}", path.display()))?
                    .len();
                if length < 20 {
                    return Err(invalid("invalid or truncated PDF upload"));
                }
                file.seek(std::io::SeekFrom::End(-1024.min(length as i64)))
                    .context("seek uploaded PDF")?;
                let mut tail = Vec::new();
                file.read_to_end(&mut tail)
                    .context("read uploaded PDF tail")?;
                tail
            }
        };
        if tail.len() < 20 || !tail.windows(5).any(|window| window == b"%%EOF") {
            return Err(invalid("invalid or truncated PDF upload"));
        }
        Ok(())
    }
}

/// Marker for an encrypted PDF offered without its password. The OCR route
/// maps this to 401 so callers can tell "supply `password`" apart from corrupt.
#[derive(Debug)]
pub struct PasswordRequired;

impl std::fmt::Display for PasswordRequired {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("encrypted PDF requires the `password` multipart field")
    }
}

impl std::error::Error for PasswordRequired {}

/// Decoding-only document facts. Runs without an engine, so metadata answers
/// work on a cold model and never enter model admission.
#[derive(Debug, Serialize)]
pub struct PageSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct DocumentMetadata {
    pub pages: usize,
    pub page_sizes: Vec<PageSize>,
}

/// Decode a PDF or supported image and report page count and rendered sizes.
/// For PDFs, sizes are at the requested DPI; an image yields one entry.
pub fn inspect_document(
    source: &DocumentSource,
    library: Option<&Path>,
    options: &PdfOptions,
) -> Result<DocumentMetadata> {
    options.validate().map_err(invalid)?;
    if source.is_pdf()? {
        source.validate_pdf_envelope()?;
        let pdfium = bind_pdfium(library)?;
        let document = open_document(&pdfium, source, options)?;
        let mut page_sizes = Vec::new();
        for page in document.pages().iter() {
            let (width, height) = page_dimensions(&page, options.dpi)?;
            page_sizes.push(PageSize { width, height });
        }
        return Ok(DocumentMetadata {
            pages: page_sizes.len(),
            page_sizes,
        });
    }
    if options.mode == OcrMode::Text {
        return Err(invalid("text mode requires a PDF text layer"));
    }
    let image = match source {
        #[cfg(test)]
        DocumentSource::Memory(data) => decode_image(data)?,
        DocumentSource::Path(path) => decode_image_file(path)?,
    };
    Ok(DocumentMetadata {
        pages: 1,
        page_sizes: vec![PageSize {
            width: image.width(),
            height: image.height(),
        }],
    })
}

/// Process a document with cancellation checkpoints between pages and image
/// regions. Byte-vector entry point for tests; the server uses the file-based
/// variants so uploads stay spooled on disk.
#[cfg(test)]
pub fn process_cancellable(
    data: &[u8],
    library: Option<&Path>,
    options: &PdfOptions,
    engine: Option<&mut Engine>,
    cancel: &AtomicBool,
) -> Result<Document> {
    let mut pages = Vec::new();
    process_source(
        &DocumentSource::Memory(data.to_vec()),
        library,
        options,
        engine,
        cancel,
        &mut |page| {
            pages.push(page);
            Ok(())
        },
    )?;
    Ok(document_from_pages(pages))
}

/// Aggregate processing of an on-disk upload (the server's non-streaming path).
pub fn process_cancellable_file(
    path: &Path,
    library: Option<&Path>,
    options: &PdfOptions,
    engine: Option<&mut Engine>,
    cancel: &AtomicBool,
) -> Result<Document> {
    let mut pages = Vec::new();
    process_source(
        &DocumentSource::Path(path.to_path_buf()),
        library,
        options,
        engine,
        cancel,
        &mut |page| {
            pages.push(page);
            Ok(())
        },
    )?;
    Ok(document_from_pages(pages))
}

/// Streaming processing of an on-disk upload: each completed page is handed
/// to the sink in page order, so peak memory stays O(one page) either way.
pub fn process_cancellable_streaming(
    path: &Path,
    library: Option<&Path>,
    options: &PdfOptions,
    engine: Option<&mut Engine>,
    cancel: &AtomicBool,
    sink: &mut dyn FnMut(Page) -> Result<()>,
) -> Result<()> {
    process_source(
        &DocumentSource::Path(path.to_path_buf()),
        library,
        options,
        engine,
        cancel,
        sink,
    )
}

fn document_from_pages(pages: Vec<Page>) -> Document {
    // 1-based inclusive bounds let chunked callers reassemble page ranges.
    let page_from = pages.first().map_or(1, |page| page.page_index + 1);
    let page_to = pages.last().map_or(0, |page| page.page_index + 1);
    Document {
        model: "datalab-to/indra-ocr@bd02167fdf2005a19962cc15188f8496a2c87d3b",
        pages,
        page_from,
        page_to,
    }
}

fn process_source(
    source: &DocumentSource,
    library: Option<&Path>,
    options: &PdfOptions,
    mut engine: Option<&mut Engine>,
    cancel: &AtomicBool,
    sink: &mut dyn FnMut(Page) -> Result<()>,
) -> Result<()> {
    options.validate().map_err(invalid)?;
    check_cancelled(cancel)?;
    if source.is_pdf()? {
        return process_pdf(source, library, options, &mut engine, cancel, sink);
    }
    if options.mode == OcrMode::Text {
        return Err(invalid("text mode requires a PDF text layer"));
    }
    let image = match source {
        #[cfg(test)]
        DocumentSource::Memory(data) => decode_image(data)?,
        DocumentSource::Path(path) => decode_image_file(path)?,
    };
    sink(required_engine(&mut engine)?.recognize_image_with_options(
        &image,
        0,
        options.keep_furniture,
    )?)
}

#[cfg(test)]
fn decode_image(data: &[u8]) -> Result<GrayImage> {
    decode_reader(std::io::Cursor::new(data))
}

fn decode_image_file(path: &Path) -> Result<GrayImage> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    decode_reader(std::io::BufReader::new(file))
}

fn decode_reader(reader: impl Read + Seek) -> Result<GrayImage> {
    let mut reader = image::ImageReader::new(std::io::BufReader::new(reader))
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

fn bind_pdfium(library: Option<&Path>) -> Result<Pdfium> {
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
    Ok(pdfium)
}

fn open_document<'a>(
    pdfium: &'a Pdfium,
    source: &DocumentSource,
    options: &PdfOptions,
) -> Result<PdfDocument<'a>> {
    // Never include the password or uploaded bytes in error context. A
    // well-formed PDF that still will not open has practically always been
    // encrypted; name the remedy so pages_only can answer 401.
    let result = match source {
        #[cfg(test)]
        DocumentSource::Memory(data) => {
            pdfium.load_pdf_from_byte_vec(data.clone(), options.password.as_deref())
        }
        DocumentSource::Path(path) => pdfium.load_pdf_from_file(path, options.password.as_deref()),
    };
    result.map_err(|error| {
        if options.password.is_none() {
            PasswordRequired.into()
        } else {
            invalid(format!("open uploaded PDF: {error}"))
        }
    })
}

fn process_pdf(
    source: &DocumentSource,
    library: Option<&Path>,
    options: &PdfOptions,
    engine: &mut Option<&mut Engine>,
    cancel: &AtomicBool,
    sink: &mut dyn FnMut(Page) -> Result<()>,
) -> Result<()> {
    let pdfium = bind_pdfium(library)?;
    let document = open_document(&pdfium, source, options)?;
    let config = PdfRenderConfig::new()
        .scale_page_by_factor(options.dpi as f32 / 72.0)
        .set_maximum_width(MAX_DIMENSION)
        .set_maximum_height(MAX_DIMENSION);
    let page_count = document.pages().len() as usize;
    // Select document page indices up front: a `pages` range must fit inside
    // the real page count, and rendering plus recognition stay strictly
    // sequential, so decoded pages never accumulate in memory.
    let indices: Vec<usize> = match &options.pages {
        Some(range) => {
            range.validate_against(page_count)?;
            (range.from..=range.to.unwrap_or(page_count))
                .map(|page| page - 1)
                .collect()
        }
        None => (0..page_count.min(options.max_pages)).collect(),
    };
    for index in indices {
        check_cancelled(cancel)?;
        let page = document
            .pages()
            .get(index.try_into().expect("page index fits PdfPageIndex"))
            .context("selected PDF page is missing")?;
        sink(process_page(
            &page, index, &config, options, engine, cancel,
        )?)?;
    }
    Ok(())
}

fn process_page(
    page: &PdfPage<'_>,
    index: usize,
    config: &PdfRenderConfig,
    options: &PdfOptions,
    engine: &mut Option<&mut Engine>,
    cancel: &AtomicBool,
) -> Result<Page> {
    // Text extraction needs the same coordinate space as OCR, without allocating
    // or painting a bitmap. These dimensions match our DPI + maximum-size config.
    let (width, height) = page_dimensions(page, options.dpi)?;
    if options.mode == OcrMode::Text {
        let lines = text_lines(page, config, width, height)?;
        return page_from_text_lines(lines, index, width, height, options.keep_furniture);
    }
    let image = render_page(page, config)?;
    if options.mode == OcrMode::Ocr {
        return required_engine(engine)?.recognize_image_with_options(
            &image,
            index,
            options.keep_furniture,
        );
    }
    let mut lines = match text_lines(page, config, image.width(), image.height()) {
        Ok(lines) => lines,
        Err(_) if options.mode == OcrMode::Auto => {
            return required_engine(engine)?.recognize_image_with_options(
                &image,
                index,
                options.keep_furniture,
            );
        }
        Err(error) => return Err(error),
    };
    if options.mode == OcrMode::Auto && !text_layer_usable(&lines, image.width(), image.height()) {
        return required_engine(engine)?.recognize_image_with_options(
            &image,
            index,
            options.keep_furniture,
        );
    }
    let mut hybrid = false;
    if options.image_ocr && options.mode != OcrMode::Text {
        let regions = image_regions(page, config, image.width(), image.height(), cancel)?;
        let text_layer_boxes = lines.iter().map(|line| line.bbox).collect::<Vec<_>>();
        if !regions.is_empty() {
            let engine = required_engine(engine)?;
            for region in regions {
                check_cancelled(cancel)?;
                let before = lines.len();
                append_image_ocr(
                    &image,
                    region,
                    &text_layer_boxes,
                    &mut lines,
                    engine,
                    index,
                    options,
                )?;
                hybrid |= lines.len() > before;
            }
        }
    }
    if hybrid {
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

fn page_dimensions(page: &PdfPage<'_>, dpi: u32) -> Result<(u32, u32)> {
    let width = page.width().value;
    let height = page.height().value;
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        bail!("PDF page has invalid dimensions");
    }
    let scale = (dpi as f32 / 72.0)
        .min(MAX_DIMENSION as f32 / width)
        .min(MAX_DIMENSION as f32 / height);
    Ok((
        (width * scale).round().max(1.0) as u32,
        (height * scale).round().max(1.0) as u32,
    ))
}

fn render_page(page: &PdfPage<'_>, config: &PdfRenderConfig) -> Result<GrayImage> {
    #[cfg(test)]
    RASTERIZATIONS.with(|count| count.set(count.get() + 1));
    Ok(page
        .render_with_config(config)
        .context("render PDF page")?
        .as_image()
        .context("decode PDF page bitmap")?
        .into_luma8())
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
        .map(|lines| group_text_segments(lines.into_iter().flatten().collect()))
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

// Preserve the text extractor's column order. Only adjacent, nearby segments
// sharing a baseline form a line; distant columns must never be joined.
fn group_text_segments(segments: Vec<Line>) -> Vec<Line> {
    let mut lines: Vec<Line> = Vec::new();
    for segment in segments {
        if let Some(previous) = lines.last_mut() {
            let a = previous.bbox;
            let b = segment.bbox;
            let height = (a[3] - a[1]).min(b[3] - b[1]);
            let overlap = a[3].min(b[3]) - a[1].max(b[1]);
            let gap = b[0] - a[2];
            if overlap >= height * 0.6 && gap >= -height * 0.25 && gap <= height * 1.5 {
                // Pdfium already supplies inferred spaces. Ink-bound gaps can
                // split ligatures such as `fi`, so preserve extracted spelling.
                previous.text.push_str(&segment.text);
                previous.bbox = [
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[2].max(b[2]),
                    a[3].max(b[3]),
                ];
                continue;
            }
        }
        lines.push(segment);
    }
    lines
}

fn visit_page_object(cancel: &AtomicBool, visited: &mut usize) -> Result<()> {
    check_cancelled(cancel)?;
    if *visited >= MAX_PAGE_OBJECTS {
        bail!("PDF page exceeds 16384 traversed objects");
    }
    *visited += 1;
    Ok(())
}

fn collect_image_rects(
    object: &PdfPageObject<'_>,
    ancestors: &mut Vec<PdfMatrix>,
    rects: &mut Vec<PdfRect>,
    cancel: &AtomicBool,
    visited: &mut usize,
) -> Result<()> {
    visit_page_object(cancel, visited)?;
    if object.as_image_object().is_some() {
        if rects.len() >= MAX_IMAGE_REGIONS {
            bail!("PDF page exceeds 256 image regions");
        }
        let rect = object.bounds()?.to_rect();
        let corners = [
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
        ];
        let mut bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        for (mut x, mut y) in corners {
            for matrix in ancestors.iter().rev() {
                (x, y) = matrix.apply_to_points(x, y);
            }
            bounds[0] = bounds[0].min(x.value);
            bounds[1] = bounds[1].min(y.value);
            bounds[2] = bounds[2].max(x.value);
            bounds[3] = bounds[3].max(y.value);
        }
        rects.push(PdfRect::new(
            PdfPoints::new(bounds[1]),
            PdfPoints::new(bounds[0]),
            PdfPoints::new(bounds[3]),
            PdfPoints::new(bounds[2]),
        ));
    } else if let Some(form) = object.as_x_object_form_object() {
        if ancestors.len() >= 16 {
            bail!("PDF image nesting exceeds supported depth");
        }
        ancestors.push(form.matrix()?);
        for index in form.as_range() {
            collect_image_rects(&form.get(index)?, ancestors, rects, cancel, visited)?;
        }
        ancestors.pop();
    }
    Ok(())
}

fn image_regions(
    page: &PdfPage<'_>,
    config: &PdfRenderConfig,
    width: u32,
    height: u32,
    cancel: &AtomicBool,
) -> Result<Vec<[f32; 4]>> {
    check_cancelled(cancel)?;
    let mut visited = 0;
    let mut rects = Vec::new();
    for object in page.objects().iter() {
        collect_image_rects(&object, &mut Vec::new(), &mut rects, cancel, &mut visited)?;
    }
    rects
        .into_iter()
        .map(|rect| rect_pixels(page, rect, config, width, height))
        .collect::<Result<Vec<_>>>()
        .map(|regions| regions.into_iter().flatten().collect())
}

// Insert within the matching column without re-sorting the extractor's lines.
fn insert_image_line(lines: &mut Vec<Line>, line: Line) {
    let position = lines
        .iter()
        .position(|prior| {
            prior.bbox[0] < line.bbox[2]
                && prior.bbox[2] > line.bbox[0]
                && prior.bbox[1] >= line.bbox[3]
        })
        .unwrap_or_else(|| {
            lines
                .iter()
                .rposition(|prior| prior.bbox[0] < line.bbox[2] && prior.bbox[2] > line.bbox[0])
                .map_or(lines.len(), |index| index + 1)
        });
    lines.insert(position, line);
}

fn append_image_ocr(
    page: &GrayImage,
    region: [f32; 4],
    text_layer_boxes: &[[f32; 4]],
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
    for (position, original) in result.lines.iter().enumerate() {
        if original.text.trim().is_empty()
            || result
                .lines
                .iter()
                .enumerate()
                .any(|(other_position, other)| {
                    position != other_position && redundant_subline(original, other)
                })
        {
            continue;
        }
        let mut line = original.clone();
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
        // The readable PDF layer owns text in its boxes. Rendered image bounds
        // can include clipped-out areas and overlaid text, and OCR spelling may
        // differ, so geometry (not string equality) decides this boundary.
        if !text_layer_boxes
            .iter()
            .any(|bbox| overlaps(*bbox, line.bbox))
            && !lines.iter().any(|prior| {
                prior.text.trim() == line.text.trim() && overlaps(prior.bbox, line.bbox)
            })
        {
            insert_image_line(lines, line);
        }
    }
    Ok(())
}

// Detectors can produce both a full line and a contained glyph crop. The latter
// adds no content when its text already occurs in the enclosing recognized line.
fn redundant_subline(line: &Line, enclosing: &Line) -> bool {
    let text = line.text.trim();
    let other = enclosing.text.trim();
    if other.len() <= text.len() || !other.contains(text) {
        return false;
    }
    let a = line.bbox;
    let b = enclosing.bbox;
    let area = (a[2] - a[0]) * (a[3] - a[1]);
    let overlap =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.0) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    area > 0.0 && overlap >= area * 0.9
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
    fn page_ranges_parse_and_name_their_bounds() {
        let single = PageRange::parse("3").unwrap();
        assert_eq!(single.from, 3);
        assert_eq!(single.to, Some(3));
        assert_eq!(PageRange::parse(" 1-10 ").unwrap().to, Some(10));
        let open = PageRange::parse("1-").unwrap();
        assert_eq!(open.to, None);
        assert_eq!(open.to_string(), "1-");
        assert_eq!(single.to_string(), "3");
        assert_eq!(PageRange::parse("4-9").unwrap().to_string(), "4-9");
        assert_eq!(single.selected_len(100), 1);
        assert_eq!(open.selected_len(7), 7);
        for garbage in ["a-b", "5-2", "0-3", "0", "", "1-2-3", "-4", "x"] {
            let error = PageRange::parse(garbage).expect_err("invalid range");
            assert!(error.to_string().contains(garbage.trim()), "{error}");
        }
        let error = PageRange::parse("5-2").unwrap_err();
        assert!(error.to_string().contains("precedes"), "{error}");
        let error = PageRange::parse("0-3").unwrap_err();
        assert!(error.to_string().contains("start at 1"), "{error}");
    }

    #[test]
    fn page_range_out_of_bounds_names_the_real_page_count() {
        let range = PageRange::parse("8-10").unwrap();
        let error = range.validate_against(7).expect_err("out of range");
        assert_eq!(
            error.to_string(),
            "document has 7 pages; pages 8-10 is out of range"
        );
        range.validate_against(10).expect("in range");
        PageRange::parse("7-")
            .unwrap()
            .validate_against(7)
            .expect("open end fits");
        let error = PageRange::parse("7-")
            .unwrap()
            .validate_against(6)
            .expect_err("open start out");
        assert_eq!(
            error.to_string(),
            "document has 6 pages; pages 7- is out of range"
        );
    }

    #[test]
    fn deadline_override_bounds_and_defaults() {
        let mut options = PdfOptions::default();
        for (value, valid) in [(29, false), (30, true), (900, true), (901, false)] {
            options.deadline_secs = Some(value);
            assert_eq!(options.validate().is_ok(), valid, "deadline_secs={value}");
        }
        options.deadline_secs = None;
        assert!(options.validate().is_ok());
    }

    #[test]
    fn image_metadata_needs_no_engine_or_pdfium() {
        let data = fixture("scan.png");
        let metadata =
            inspect_document(&DocumentSource::Memory(data), None, &PdfOptions::default())
                .expect("image metadata");
        assert_eq!(metadata.pages, 1);
        assert_eq!(
            (metadata.page_sizes[0].width, metadata.page_sizes[0].height),
            (1653, 2337)
        );
        let options = PdfOptions {
            mode: OcrMode::Text,
            ..PdfOptions::default()
        };
        let error = inspect_document(
            &DocumentSource::Memory(b"not-a-pdf".to_vec()),
            None,
            &options,
        )
        .expect_err("text mode rejects images");
        assert!(error.to_string().contains("requires a PDF"));
    }

    #[test]
    fn malformed_input_is_typed_and_cancellation_precedes_parsing() {
        let options = PdfOptions::default();
        for bytes in [&b"corrupt image"[..], &b"%PDF-1.7 truncated"[..]] {
            let error = inspect_document(&DocumentSource::Memory(bytes.to_vec()), None, &options)
                .expect_err("invalid input");
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
        let before = RASTERIZATIONS.with(std::cell::Cell::get);
        let first = process(&text_pdf(), Some(Path::new(&library)), &options, None).unwrap();
        assert_eq!(
            RASTERIZATIONS.with(std::cell::Cell::get),
            before,
            "text mode must not rasterize the page"
        );
        assert_eq!(first.pages.len(), 1);
        assert_eq!(first.pages[0].source, "text");
        assert!(first.pages[0].text.contains("without model weights"));
        options.dpi = 144;
        let second = process(&text_pdf(), Some(Path::new(&library)), &options, None).unwrap();
        assert!(
            (second.pages[0].lines[0].bbox[0] - 2.0 * first.pages[0].lines[0].bbox[0]).abs() <= 2.0
        );
    }
    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(name),
        )
        .unwrap()
    }

    fn line(text: &str, bbox: [f32; 4]) -> Line {
        Line {
            text: text.to_owned(),
            bbox,
            confidence: 1.0,
        }
    }

    #[test]
    fn styled_segments_join_without_merging_distant_columns_or_rows() {
        let lines = group_text_segments(vec![
            line("Styled", [10.0, 10.0, 50.0, 20.0]),
            line(" words", [50.0, 10.0, 90.0, 20.0]),
            line("right column", [200.0, 10.0, 290.0, 20.0]),
            line("next row", [10.0, 30.0, 90.0, 40.0]),
        ]);
        assert_eq!(
            lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["Styled words", "right column", "next row"]
        );
        assert_eq!(lines[0].bbox, [10.0, 10.0, 90.0, 20.0]);
    }

    #[test]
    fn ink_gaps_inside_words_do_not_invent_spaces() {
        let lines = group_text_segments(vec![
            line("fi", [10.0, 10.0, 20.0, 20.0]),
            line("rst", [22.0, 10.0, 40.0, 20.0]),
        ]);
        assert_eq!(lines[0].text, "first");
    }

    #[test]
    fn duplicate_glyph_crops_are_removed_only_inside_matching_text() {
        let enclosing = line("IMAGE OCR WORDS", [10.0, 10.0, 100.0, 30.0]);
        assert!(redundant_subline(
            &line("D", [70.0, 10.0, 80.0, 30.2]),
            &enclosing
        ));
        assert!(!redundant_subline(
            &line("D", [70.0, 50.0, 80.0, 60.0]),
            &enclosing
        ));
        assert!(!redundant_subline(
            &line("X", [70.0, 10.0, 80.0, 30.0]),
            &enclosing
        ));
    }

    #[test]
    fn image_text_insertion_preserves_column_reading_order() {
        let mut lines = vec![
            line("left top", [10.0, 10.0, 90.0, 20.0]),
            line("left bottom", [10.0, 80.0, 90.0, 90.0]),
            line("right top", [200.0, 10.0, 290.0, 20.0]),
            line("right bottom", [200.0, 80.0, 290.0, 90.0]),
        ];
        insert_image_line(&mut lines, line("left image", [10.0, 40.0, 90.0, 60.0]));
        assert_eq!(
            lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            [
                "left top",
                "left image",
                "left bottom",
                "right top",
                "right bottom"
            ]
        );
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY; page ranges and metadata inspect a real PDF without weights"]
    fn real_pdf_page_range_and_metadata_use_the_document_page_count() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let data = fixture("scan.pdf");
        let options = PdfOptions {
            pages: Some(PageRange::parse("2-").unwrap()),
            ..PdfOptions::default()
        };
        let error = process(&data, Some(Path::new(&library)), &options, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("document has 1 pages; pages 2- is out of range"),
            "{error}"
        );
        let metadata = inspect_document(
            &DocumentSource::Memory(data),
            Some(Path::new(&library)),
            &PdfOptions::default(),
        )
        .unwrap();
        assert_eq!(metadata.pages, 1);
        assert!(metadata.page_sizes[0].width > 0 && metadata.page_sizes[0].height > 0);
        // dpi scales the reported rendered size.
        let options = PdfOptions {
            dpi: 72,
            ..PdfOptions::default()
        };
        let low = inspect_document(
            &DocumentSource::Memory(fixture("scan.pdf")),
            Some(Path::new(&library)),
            &options,
        )
        .unwrap();
        assert!(low.page_sizes[0].width < metadata.page_sizes[0].width);
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY; encrypted PDFs name the missing password in metadata mode"]
    fn real_encrypted_pdf_metadata_requires_a_password() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let data = fixture("encrypted.pdf");
        let error = inspect_document(
            &DocumentSource::Memory(data),
            Some(Path::new(&library)),
            &PdfOptions::default(),
        )
        .unwrap_err();
        assert!(error.is::<PasswordRequired>(), "{error}");
        assert!(error.to_string().contains("password"), "{error}");
        // The same call with the fixture password decodes without an engine.
        let options = PdfOptions {
            password: Some("fixture-password".into()),
            ..PdfOptions::default()
        };
        let metadata = inspect_document(
            &DocumentSource::Memory(fixture("encrypted.pdf")),
            Some(Path::new(&library)),
            &options,
        )
        .unwrap();
        assert_eq!(metadata.pages, 1);
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY; opens a real password-protected PDF"]
    fn real_encrypted_pdf_requires_the_correct_password() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let data = fixture("encrypted.pdf");
        let mut options = PdfOptions {
            mode: OcrMode::Text,
            ..PdfOptions::default()
        };
        for password in [None, Some("incorrect-password")] {
            options.password = password.map(str::to_owned);
            let error = process(&data, Some(Path::new(&library)), &options, None).unwrap_err();
            assert!(matches!(
                error.downcast_ref::<InputError>(),
                Some(InputError::Invalid(_))
            ));
            assert!(!error.to_string().contains("incorrect-password"));
        }
        options.password = Some("fixture-password".to_owned());
        let document = process(&data, Some(Path::new(&library)), &options, None).unwrap();
        assert_eq!(document.pages[0].source, "text");
        assert!(document.pages[0].text.contains("embedded blank image"));
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY; exercises image-containing PDFs without ONNX sessions"]
    fn real_text_mode_ignores_images_and_groups_styled_segments() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        // image_ocr deliberately remains true: explicit text mode must never need an engine.
        let options = PdfOptions {
            mode: OcrMode::Text,
            dpi: 72,
            ..PdfOptions::default()
        };
        let document = process(
            &fixture("text-with-image.pdf"),
            Some(Path::new(&library)),
            &options,
            None,
        )
        .unwrap();
        assert_eq!(document.pages[0].source, "text");
        assert!(document.pages[0].text.contains("embedded blank image"));
        let styled = process(
            &fixture("text-segments.pdf"),
            Some(Path::new(&library)),
            &options,
            None,
        )
        .unwrap();
        assert_eq!(
            styled.pages[0].lines.len(),
            3,
            "{:?}",
            styled.pages[0].lines
        );
        assert!(styled.pages[0].lines[0].text.contains("Styled"));
        assert!(styled.pages[0].lines[0].text.contains("words"));
        assert!(!styled.pages[0].lines[0].text.contains("column"));
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY and ONNX_OCR_MODEL_DIR; real hybrid and blank-image OCR"]
    fn real_hybrid_pdf_adds_image_words_without_reordering_text_columns() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let directory = std::env::var("ONNX_OCR_MODEL_DIR").expect("ONNX_OCR_MODEL_DIR");
        let mut engine = Engine::load(Path::new(&directory)).unwrap();
        let mut options = PdfOptions::default();
        let blank = process(
            &fixture("text-with-image.pdf"),
            Some(Path::new(&library)),
            &options,
            Some(&mut engine),
        )
        .unwrap();
        assert_eq!(
            blank.pages[0].source, "text",
            "a blank image adds no OCR text"
        );
        options.mode = OcrMode::Text;
        let baseline = process(
            &fixture("hybrid.pdf"),
            Some(Path::new(&library)),
            &options,
            None,
        )
        .unwrap();
        options.mode = OcrMode::Auto;
        let hybrid = process(
            &fixture("hybrid.pdf"),
            Some(Path::new(&library)),
            &options,
            Some(&mut engine),
        )
        .unwrap();
        assert_eq!(hybrid.pages[0].source, "hybrid");
        assert!(
            hybrid.pages[0].text.contains("IMAGE OCR WORDS"),
            "{}",
            hybrid.pages[0].text
        );
        assert_eq!(
            hybrid.pages[0].lines.len(),
            baseline.pages[0].lines.len() + 1,
            "{:?}",
            hybrid.pages[0].lines
        );
        let preserved = hybrid.pages[0]
            .lines
            .iter()
            .filter(|line| {
                baseline.pages[0]
                    .lines
                    .iter()
                    .any(|original| original.text == line.text && original.bbox == line.bbox)
            })
            .map(|line| line.text.trim())
            .collect::<Vec<_>>();
        let expected = baseline.pages[0]
            .lines
            .iter()
            .map(|line| line.text.trim())
            .collect::<Vec<_>>();
        assert_eq!(
            preserved, expected,
            "adding image OCR must not interleave PDF text columns"
        );
        let image = hybrid.pages[0]
            .lines
            .iter()
            .find(|line| line.text.contains("IMAGE OCR WORDS"))
            .unwrap();
        assert!(
            image.bbox[0] >= 100.0 && image.bbox[2] <= 840.0,
            "{:?}",
            image.bbox
        );
    }

    fn pdf_objects(objects: Vec<Vec<u8>>) -> Vec<u8> {
        use std::io::Write;
        let mut output = b"%PDF-1.7\n".to_vec();
        let mut offsets = vec![0];
        for (index, object) in objects.iter().enumerate() {
            offsets.push(output.len());
            writeln!(output, "{} 0 obj", index + 1).unwrap();
            output.extend_from_slice(object);
            output.extend_from_slice(b"\nendobj\n");
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
        output
    }

    fn stream(dictionary: &str, bytes: &[u8]) -> Vec<u8> {
        let mut output =
            format!("<< {dictionary} /Length {} >>\nstream\n", bytes.len()).into_bytes();
        output.extend_from_slice(bytes);
        output.extend_from_slice(b"\nendstream");
        output
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY; inspects image bounds inside two transformed PDF forms"]
    fn real_nested_form_extracts_image_bounds_instead_of_surrounding_text() {
        let data = pdf_objects(vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /XObject << /Fm 4 0 R >> >> /Contents 5 0 R >>".to_vec(),
            stream("/Type /XObject /Subtype /Form /BBox [0 0 200 200] /Resources << /XObject << /Inner 6 0 R >> /Font << /F1 8 0 R >> >>",
                b"BT /F1 12 Tf 0 180 Td (Form text must stay outside the image region) Tj ET q 1 0 0 1 20 30 cm /Inner Do Q"),
            stream("", b"q 1 0 0 1 50 60 cm /Fm Do Q"),
            stream("/Type /XObject /Subtype /Form /BBox [0 0 100 100] /Resources << /XObject << /Im 7 0 R >> >>",
                b"q 40 0 0 20 10 15 cm /Im Do Q"),
            stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8", &[255]),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ]);
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let lock = PDFIUM_BIND_LOCK.lock().unwrap();
        let pdfium = match Pdfium::bind_to_library(Path::new(&library)) {
            Ok(bindings) => Pdfium::new(bindings),
            Err(PdfiumError::PdfiumLibraryBindingsAlreadyInitialized) => Pdfium::default(),
            Err(error) => panic!("load Pdfium: {error}"),
        };
        drop(lock);
        let document = pdfium.load_pdf_from_byte_vec(data, None).unwrap();
        let page = document.pages().get(0).unwrap();
        let regions = image_regions(
            &page,
            &PdfRenderConfig::new(),
            300,
            300,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(regions.len(), 1);
        assert!(
            image_regions(
                &page,
                &PdfRenderConfig::new(),
                300,
                300,
                &AtomicBool::new(true)
            )
            .unwrap_err()
            .to_string()
            .contains("cancelled")
        );
        let object = page.objects().get(0).unwrap();
        let mut visited = MAX_PAGE_OBJECTS - 1;
        let error = collect_image_rects(
            &object,
            &mut Vec::new(),
            &mut Vec::new(),
            &AtomicBool::new(false),
            &mut visited,
        )
        .unwrap_err();
        assert!(error.to_string().contains("traversed objects"));
        assert_eq!(
            visited, MAX_PAGE_OBJECTS,
            "nested traversal must share the outer budget"
        );
        for (actual, expected) in regions[0].iter().zip([80.0, 175.0, 120.0, 195.0]) {
            assert!(
                (actual - expected).abs() <= 1.0,
                "actual image bounds: {:?}",
                regions[0]
            );
        }
    }
    #[test]
    fn object_traversal_checks_cancellation_and_bounds_visits_without_images() {
        let cancel = AtomicBool::new(false);
        let mut visited = 0;
        for _ in 0..MAX_PAGE_OBJECTS {
            visit_page_object(&cancel, &mut visited).unwrap();
        }
        assert!(
            visit_page_object(&cancel, &mut visited)
                .unwrap_err()
                .to_string()
                .contains("traversed objects")
        );
        assert_eq!(visited, MAX_PAGE_OBJECTS);
        cancel.store(true, Ordering::Relaxed);
        let mut visited = 0;
        assert!(
            visit_page_object(&cancel, &mut visited)
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
        assert_eq!(visited, 0);
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY; compares text geometry dimensions against actual capped rendering"]
    fn real_text_dimensions_match_rotated_and_capped_bitmaps_without_rendering() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let lock = PDFIUM_BIND_LOCK.lock().unwrap();
        let pdfium = match Pdfium::bind_to_library(Path::new(&library)) {
            Ok(bindings) => Pdfium::new(bindings),
            Err(PdfiumError::PdfiumLibraryBindingsAlreadyInitialized) => Pdfium::default(),
            Err(error) => panic!("load Pdfium: {error}"),
        };
        drop(lock);
        let document = pdfium.load_pdf_from_byte_vec(text_pdf(), None).unwrap();
        let mut page = document.pages().get(0).unwrap();
        for rotation in [
            PdfPageRenderRotation::None,
            PdfPageRenderRotation::Degrees90,
        ] {
            page.set_rotation(rotation);
            for dpi in [72, 144, 600] {
                let config = PdfRenderConfig::new()
                    .scale_page_by_factor(dpi as f32 / 72.0)
                    .set_maximum_width(MAX_DIMENSION)
                    .set_maximum_height(MAX_DIMENSION);
                let dimensions = page_dimensions(&page, dpi).unwrap();
                let bitmap = page.render_with_config(&config).unwrap();
                assert_eq!(dimensions, (bitmap.width() as u32, bitmap.height() as u32));
                let before = RASTERIZATIONS.with(std::cell::Cell::get);
                let options = PdfOptions {
                    dpi,
                    mode: OcrMode::Text,
                    ..PdfOptions::default()
                };
                let text = process_page(
                    &page,
                    0,
                    &config,
                    &options,
                    &mut None,
                    &AtomicBool::new(false),
                )
                .unwrap();
                assert_eq!(RASTERIZATIONS.with(std::cell::Cell::get), before);
                assert!(text.text.contains("without model weights"));
            }
        }
    }

    // A white image is clipped to the bottom of the page, but its un-clipped
    // object bounds encompass the PDF text above it. A ToUnicode mapping makes
    // extracted spelling differ intentionally from the painted glyphs, proving
    // that comparing OCR and text strings cannot prevent duplicate extraction.
    fn clipped_image_over_text_pdf() -> Vec<u8> {
        use std::fmt::Write;
        let mut mapping = "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /FixtureUnicode def\n/CMapType 2 def\n1 begincodespacerange\n<00> <FF>\nendcodespacerange\n95 beginbfchar\n".to_owned();
        for character in 32..=126u8 {
            let mapped = if character == b'R' { b'X' } else { character };
            writeln!(mapping, "<{character:02X}> <{mapped:04X}>").unwrap();
        }
        mapping.push_str(
            "endbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend",
        );
        pdf_objects(vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 400] /Resources << /XObject << /Im 6 0 R >> /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_vec(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 7 0 R >>".to_vec(),
            stream("", b"q 40 40 500 60 re W n 500 0 0 300 40 40 cm /Im Do Q BT /F1 18 Tf 50 250 Td (RENDERED WORDS FROM THE PDF TEXT LAYER) Tj ET"),
            stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8", &[255]),
            stream("", mapping.as_bytes()),
        ])
    }

    #[test]
    #[ignore = "requires PDFIUM_LIBRARY and ONNX_OCR_MODEL_DIR; verifies clipped image bounds cannot duplicate authoritative text"]
    fn real_clipped_image_does_not_duplicate_differently_spelled_pdf_text() {
        let library = std::env::var("PDFIUM_LIBRARY").expect("PDFIUM_LIBRARY");
        let directory = std::env::var("ONNX_OCR_MODEL_DIR").expect("ONNX_OCR_MODEL_DIR");
        let data = clipped_image_over_text_pdf();
        let mut engine = Engine::load(Path::new(&directory)).unwrap();
        let mut options = PdfOptions {
            mode: OcrMode::Ocr,
            ..PdfOptions::default()
        };
        let raster = process(
            &data,
            Some(Path::new(&library)),
            &options,
            Some(&mut engine),
        )
        .unwrap();
        assert!(
            raster.pages[0].text.contains("RENDERED WORDS"),
            "{}",
            raster.pages[0].text
        );
        options.mode = OcrMode::Auto;
        let text = process(
            &data,
            Some(Path::new(&library)),
            &options,
            Some(&mut engine),
        )
        .unwrap();
        assert!(
            text.pages[0].text.contains("XENDEXED WOXDS"),
            "{}",
            text.pages[0].text
        );
        assert!(!text.pages[0].text.contains("RENDERED"));
        assert_eq!(
            text.pages[0].source, "text",
            "overlapping OCR adds no content"
        );
        assert_eq!(text.pages[0].lines.len(), 1, "{:?}", text.pages[0].lines);
    }
}
