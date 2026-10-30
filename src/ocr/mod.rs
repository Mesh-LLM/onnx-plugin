//! Native Indra model execution and document assembly.

mod detection;
pub mod document;
mod layout;
mod math;
mod orientation;
mod recognition;

use std::path::Path;

use anyhow::{Context, Result, bail};
use image::GrayImage;
use ort::session::Session;
use serde::Serialize;

use self::detection::detect_lines;
use self::layout::{Block, assemble};
use self::orientation::orient;
use self::recognition::Recognizers;

const MODEL: &str = "datalab-to/indra-ocr@v1.1.0";

#[derive(Debug, Clone, Serialize)]
pub struct Line {
    pub text: String,
    pub bbox: [f32; 4],
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Page {
    pub page_index: usize,
    pub rotation: u16,
    pub vertical: bool,
    pub text: String,
    pub markdown: String,
    pub source: &'static str,
    pub lines: Vec<Line>,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Document {
    pub model: &'static str,
    pub pages: Vec<Page>,
}

pub struct Engine {
    detector: Session,
    recognizers: Recognizers,
    order: Option<Session>,
    orientation: Option<Session>,
    det_long_side: usize,
    math: Option<math::MathReader>,
}

impl Engine {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Self::load_with_options(model_dir, 4, false)
    }

    /// Load the pinned Indra pipeline with a bounded CPU thread count.
    ///
    /// Math additionally loads the classifier and cached LaTeX decoder graphs.
    pub fn load_with_options(model_dir: &Path, threads: usize, math: bool) -> Result<Self> {
        validate_load_options(threads, math)?;
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("meta.json")).context("read Indra meta.json")?,
        )?;
        let det_long_side = metadata["det_long_side"]
            .as_u64()
            .context("Indra det_long_side missing")? as usize;
        let math = math
            .then(|| {
                let height = metadata["rec_height"]
                    .as_u64()
                    .context("Indra rec_height missing")? as usize;
                math::MathReader::load(model_dir, threads, height)
            })
            .transpose()?;
        let detector = session_with_threads(model_dir.join("det.onnx"), threads)?;
        let recognizers = Recognizers::load(model_dir, &metadata, threads)?;
        let order = model_dir.join("order.onnx");
        let order = order
            .is_file()
            .then(|| session_with_threads(order, threads))
            .transpose()?;
        let orientation = model_dir.join("orient.onnx");
        let orientation = orientation
            .is_file()
            .then(|| session_with_threads(orientation, threads))
            .transpose()?;
        Ok(Self {
            detector,
            recognizers,
            order,
            orientation,
            det_long_side,
            math,
        })
    }

    pub fn recognize_image(&mut self, image: &GrayImage, page_index: usize) -> Result<Page> {
        self.recognize_image_with_options(image, page_index, false)
    }

    /// Recognize one image, retaining structural headers/footers in JSON while
    /// independently controlling their inclusion in Markdown.
    pub fn recognize_image_with_options(
        &mut self,
        image: &GrayImage,
        page_index: usize,
        keep_furniture: bool,
    ) -> Result<Page> {
        if image.width() == 0 || image.height() == 0 {
            bail!("document page is empty");
        }
        let upright = orient(self.orientation.as_mut(), image)?;
        let image = &upright.image;
        let boxes = detect_lines(&mut self.detector, image, self.det_long_side)?;
        let mut lines = self.recognizers.recognize_lines(image, &boxes)?;
        if let Some(math) = self.math.as_mut() {
            math.recognize(image, &mut lines)?;
        }
        let layout = assemble(self.order.as_mut(), lines, image.width(), image.height())?;
        let lines = layout.lines;
        let text = lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Page {
            page_index,
            rotation: upright.rotation,
            vertical: upright.vertical,
            markdown: if keep_furniture {
                layout::markdown(&layout.blocks, true)
            } else {
                layout.markdown
            },
            text,
            source: "ocr",
            lines,
            blocks: layout.blocks,
        })
    }

    pub fn recognize_document(
        &mut self,
        pages: impl IntoIterator<Item = Result<GrayImage>>,
    ) -> Result<Document> {
        let pages = pages
            .into_iter()
            .enumerate()
            .map(|(index, image)| self.recognize_image(&image?, index))
            .collect::<Result<Vec<_>>>()?;
        Ok(Document {
            model: MODEL,
            pages,
        })
    }
}

pub fn validate_load_options(threads: usize, _math: bool) -> Result<()> {
    if !(1..=256).contains(&threads) {
        bail!("Indra threads must be between 1 and 256");
    }
    Ok(())
}

fn session_with_threads(path: impl AsRef<Path>, threads: usize) -> Result<Session> {
    let path = path.as_ref();
    if !path.is_file() {
        bail!("Indra model file missing: {}", path.display());
    }
    Ok(Session::builder()?
        .with_intra_threads(threads)
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .with_memory_pattern(false)
        .map_err(|error| anyhow::anyhow!("{error}"))?
        .commit_from_file(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_options_fail_before_reading_assets() {
        let absent = Path::new("/not-an-indra-model");
        assert!(
            Engine::load_with_options(absent, 0, false)
                .err()
                .expect("invalid threads")
                .to_string()
                .contains("threads")
        );
        assert!(validate_load_options(4, true).is_ok());
        assert!(validate_load_options(256, false).is_ok());
        assert!(validate_load_options(257, false).is_err());
    }

    #[test]
    #[ignore = "requires Indra v1.1.0 model assets"]
    fn indra_sample_runs_through_native_ort() {
        let model_dir = std::env::var("ONNX_OCR_MODEL_DIR").expect("ONNX_OCR_MODEL_DIR");
        let mut engine = Engine::load(Path::new(&model_dir)).expect("load ONNX sessions");
        let image = image::open(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/scan.png"
        ))
        .expect("open sample")
        .into_luma8();
        let page = engine.recognize_image(&image, 0).expect("recognize sample");
        assert_eq!(page.lines.len(), 35, "native line detection changed");
        assert_eq!(page.blocks.len(), 8, "native layout grouping changed");
        let heading = [
            "대한성공회 서울교구",
            "교구장 이경호 주교",
            "04519 서울특별시 중구 세종대로 21길 15",
            "전화 (02)735-6157 / 팩스 (02)723-2640",
            "DIOCESE OF SEOUL",
            "THE ANGLICAN CHURCH OF KOREA",
        ];
        assert_eq!(
            page.lines
                .iter()
                .take(6)
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            heading
        );
        assert!(page.text.contains("온라인 전례 실황"), "body text missing");
        assert!(page.markdown.contains("대한성공회 서울교구"));
    }
}
